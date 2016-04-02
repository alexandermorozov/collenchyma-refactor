/// This module is an attempt to figure out how to improve structure of
/// `collenchyma`.
///
/// Complete:
/// - Allow to separate CUDA, OpenCL and Native backends into their own
///   independent crates. No need to specify features during compilation,
///   just import `collenchyma-cuda`, `collenchyma-opencl` as required.
///   This doesn't come for free, though.
/// - Implement reasonable protection from use of unitialized memory in runtime.
/// - Don't do synchronization when it's not strictly required. E.g. if copies
///   on both GPU and CPU are both current, getting read-only references to
///   any of them won't result in synchronization.
/// - Allow to skip initialization for write-only buffers.
/// - Restrict memory mutability in common Rust style: allow only one mutable
///   "view" (implemeted as `Tensor`), or many immutable views.
/// - Use type system and borrowck to insure that tensors are synchronized
///   properly.
/// - Allow syncs and mem allocations on immutable `SharedTensor`. This allows
///   to place syncs() closer to use sites, see leaf::layer::sync().
/// - All computations are done on `Tensor` and `MutTensor`. `Tensor` backed by
///   immutable memory and `MutTensor` is backed by mutable memory. Mutability
///   of contents are enforced by Rust's type system.
/// - `Tensor` and `MutTensor` are parametrized on backend driver and give out
///   typed memory on request, so there is no need to use .as_native(), etc.
/// - `Tensor` and `MutTensor` can be reshaped in-place. Reshaping doesn't
///   affect their parent `SharedTensor`.
///
/// Possible:
/// - It shouldn't be difficult to implement slicing, but that would probably
///   require implementing slicing for memory first. Slicing can be used to cut
///   number of memory transfers. E.g. solver may need to update scalars for
///   local learning rate on GPU for each of `N` weight tensors every epoch.
///   That'll result in `N` 4 byte transfers from host to GPU that could be
///   replaced with single transfer with slicing.
/// - Async operations: it looks like currently most time is spent waiting
///   for in/out transfers even on mid-range GPU hardware. Async may help a lot.
///   Async can be implemented by making transfer_to/transfer_from to return
///   an object that can be waited on until transfer completes. Later
///   `Tensor::get_memory()` could block until transfer completes when
///   direction is e.g. CUDA -> Host.
///
/// Tradeoffs:
/// - Moving CUDA and OpenCL stuff completely out of the base crate would remove
///   conditional compilation (it's evil in general), remove several enums,
///   wrapping and unwrapping throughout code. On the other side it complicates
///   interoperation: CUDA backend must know specifics of Native to implement
///   memory synchronization. I haven't found a way to do this without using
///   `Any` for dynamic-like typing, and that's quite ugly and wastes extra
///   CPU cycles. I think added delay is negligible in comparison with
///   setting up any kind of Host <--> discrete GPU transfer. That said, syncs()
///   should be very rare: initialize, then upload NN input data and download
///   output.
/// - tensor objects need to have two types of mutability: one to allow reshaping,
///   another to indicate if underlying mem is mutable. Currently former is
///   achived with Rust's usual mutability rules, and latter is implemented with
///   split into `Tensor` and `MutTensor`. I feel like `Tensor` and `MutTensor`
///   can be combined if e.g. `SharedTensor::read()` returns `&Tensor` instead
///   of `Tensor`. In order to do that, `SharedTensor` must own Tensors.
///   And `Tensor::reshape()` will have to return a new object that again
///   is owned by the same parent `SharedTensor`... It may or may not be
///   possible to write code like this with good user interface, that
///   doesn't allocate each time and doesn't use additional traits and
///   inderection.

use std::any::Any;
use std::borrow::{Cow, Borrow};
use std::cell::{Ref, RefMut, RefCell};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;


/// This trait should be implemented for `Device`.
/// Use of `Any` everywhere is ugly, but it looks like there is no other way
/// to do it if we want to extract CUDA stuff into its own crate completely,
/// so that base crate knows nothing about it at all.
pub trait MemoryTransfer {
    fn can_transfer_from(&self, src_device: &Any) -> bool;
    fn can_transfer_to(&self, dst_device: &Any) -> bool;

    fn transfer_from(&self, my_memory: &Any, dst_device: &Any, dst_memory: &Any)
                     -> Result<(), Error>;
    fn transfer_to(&self, my_memory: &Any, src_device: &Any, src_memory: &Any)
                   -> Result<(), Error>;
}

pub trait Device
    where Self: Clone + Eq + Any + Debug {

    type M: Any;
    fn allocate_memory(&Self, usize) -> Result<Self::M, Error>;
}



/// Stub for `Error` type.
#[derive(Debug)]
pub enum Error {
    UninitializedMemory,
    AllocationFailed,
    NoMemoryTransferRoute,
}

/// `Tensor` located on a specific device.
/// It's possible to extend it to allow slicing if slice memory is continuous.
/// TODO: define methods like reshape(), get_mem(), etc.
pub struct Tensor<'a, D: Device> {
    dim: Cow<'a, [usize]>,
    device: D,
    memory: Ref<'a, D::M>,
}

/// `MutTensor` located on a specific device.
/// It's possible to extend it to allow slicing if slice memory is continuous.
/// TODO: define methods like reshape(), get_mem(), .get_mut_mem(), .fill(), etc.
pub struct MutTensor<'a, D: Device> {
    dim: Cow<'a, [usize]>,
    device: D,
    memory: RefMut<'a, D::M>,
}

/// Unsigned integer type that keeps version of memory location.
/// Each time when ``Tensor` is mutably borrowed from `SharedTensor`, version
/// of corresponding memory is increased.
/// Value `0` has special meaning: it means that memory location wasn't
/// initialized yet.
type Version = u32;

/// Helper type that keeps full description of memory location.
/// It's not part of API.
struct TensorLocation {
    device: Box<Any>,
    version: Version,
    mem: Rc<Any>,
}

/// `SharedTensor` keeps track of all memory locations and their versions
/// and does memory transfers when they are required.
pub struct SharedTensor {
    dim: Vec<usize>,

    locations: RefCell<Vec<TensorLocation>>,

    latest_version: Version,
}

impl SharedTensor {
    pub fn new(dim: Vec<usize>) -> SharedTensor {
        SharedTensor {
            dim: dim,
            locations: RefCell::new(Vec::new()),
            latest_version: 0,
        }
    }

    /// I suggest putting all `collenchyma::TensorDesc` methods directly on
    /// `SharedTensor` and `Tensor`. Doing `x.desc().size()` is overly verbose.
    /// This can be implemented via additional trait or as macros.
    pub fn size(&self) -> usize {
        self.dim.iter().fold(1, |s, &x| s * x)
    }

    /// Looks up `device` in self.locations and returns its index. If lookup
    /// fails then new location is created and its index is returned.
    fn get_location_index<D: Device>(&self, device: &D)
                                     -> Result<usize, Error> {
        for (i, loc) in self.locations.borrow().iter().enumerate() {
            match loc.device.deref().downcast_ref::<D>() {
                Some(ref d) if *d == device => return Ok(i),
                _ => {}
                // Some(ref d) => {
                //     if *d == device {
                //         return Ok(i);
                //     } else {
                //         println!("devs differ: {:?} != {:?}", *d, device);
                //     }
                // },
                // _ => {
                //     println!("downcast failed!");
                // },
            }
        }
        println!("Add new device");
        self.locations.borrow_mut().push(TensorLocation {
            device: Box::new(device.clone()),
            version: 0,
            mem: Rc::new(try!(D::allocate_memory(device, self.size()))),
        });
        Ok(self.locations.borrow().len() - 1)
    }

    /// TODO: chose the best source to copy data from.
    /// That would require some additional traits that return costs for
    /// transferring data between different backends.
    /// Actually I think that there would be only transfers between
    /// `Native` <-> `Cuda` and `Native` <-> `OpenCL` in foreseeable future,
    /// so it's best to not overengineer here.
    fn update_if_needed<D: Device>(&self, device: &D, dst_i: usize)
                                   -> Result<(), Error> {
        let locs = self.locations.borrow_mut();
        let dst_loc = &locs[dst_i];
        if dst_loc.version == self.latest_version {
            return Ok(());
        }

        // TODO: filter out impossible transfers
        let (src_i, src_loc) = locs.iter().enumerate()
            .filter(|&(_, loc)| loc.version == self.latest_version)
            .next().expect("broken invariant: can't find latest version");

        println!("src_i={} dst_i={}", src_i, dst_i);
        if src_i == dst_i {
            return Ok(());
        }

        let src_device = src_loc.device.deref().downcast_ref::<&MemoryTransfer>()
            .expect("broken invariant: object doesn't support mem transfers");
        let dst_device = dst_loc.device.deref().downcast_ref::<&MemoryTransfer>()
            .expect("broken invariant: object doesn't support mem transfers");

        if src_device.can_transfer_to(&dst_loc.device) {
            src_device.transfer_to(&src_loc.mem, &dst_loc.device, &dst_loc.mem)
        } else if src_device.can_transfer_to(&dst_loc.device) {
            dst_device.transfer_from(&dst_loc.device, &src_loc.device, &src_loc.mem)
        } else {
            // TODO: fallback on indirect transfer via Native
            Err(Error::NoMemoryTransferRoute)
        }
    }

    /// Get tensor for reading on the specified `device`.
    /// Can fail if memory allocation fails, or tensor wasn't initialized yet.
    pub fn read<'a, D: Device>(&'a self, device: &D)
                               -> Result<Tensor<'a, D>, Error> {
        if self.latest_version == 0 {
            return Err(Error::UninitializedMemory);
        }
        let i = try!(self.get_location_index::<D>(device));
        try!(self.update_if_needed::<D>(device, i));

        let locs = self.locations.borrow();
        Ok(Tensor {
            dim: Cow::Borrowed(&self.dim),
            memory: Ref::map(locs,
                             |ls| ls[i].mem.deref().downcast_ref::<D::M>()
                             .expect("Wrong mem type")),
            device: device.clone(),
        })
    }

    /// Get tensor for reading and writing on the specified `device`.
    /// This memory location is set as latest.
    /// Can fail if memory allocation fails, or tensor wasn't initialized yet.
    pub fn read_write<'a, D: Device>(&'a self, device: &D)
                                     -> Result<Tensor<'a, D>, Error> {
        unimplemented!();
    }

    /// Get tensor for writing only.
    /// This function skips synchronization and initialization logic, since
    /// contents will be overwritten anyway. Caller must initialize all elements
    /// of this tensor. This convention isn't enforced, and failure to do so
    /// may result in use of undefined data later.
    /// If caller has failed to overwrite memory, it must call `invalidate()`
    /// to return vector to uninitialized state.
    pub fn write_only<'a, D: Device>(&'a mut self, device: &D)
                                     -> Result<MutTensor<'a, D>, Error> {
        let i = try!(self.get_location_index::<D>(device));

        let mut locs = self.locations.borrow_mut();

        // FIXME: properly wrap versions on overflow
        self.latest_version += 1;
        locs[i].version += self.latest_version;

        Ok(MutTensor {
            dim: Cow::Borrowed(&self.dim),
            memory: RefMut::map(locs,
                                |ls| Rc::get_mut(&mut ls[i].mem).unwrap()
                                .downcast_mut::<D::M>()
                                .expect("Wrong mem type")),
            device: device.clone(),
        })
    }

    /// Invalidate data at all memory locations. This function only marks
    /// memory as invalid and doen't deallocate anything.
    pub fn invalidate(&mut self) {
        unimplemented!();
    }
}


#[cfg(test)]
mod tests {
    use std::any::Any;
    use super::*;

    #[derive(PartialEq, Eq, Clone, Debug)]
    struct CpuDevice {
        index: usize,
    }

    struct HostMemory {
        data: Vec<u8>
    }

    impl Device for CpuDevice {
        type M = HostMemory;

        fn allocate_memory(_dev: &Self, size: usize) -> Result<Self::M, Error> {
            Ok(HostMemory {
                data: vec![0; size]
            })
        }
    }

    impl MemoryTransfer for CpuDevice {
        fn can_transfer_from(&self, _src_device: &Any) -> bool {
            false
        }

        fn can_transfer_to(&self, _dst_device: &Any) -> bool {
            false
        }

        fn transfer_from(&self, _my_memory: &Any, _dst_device: &Any, _dst_memory: &Any)
                         -> Result<(), Error> {
            unimplemented!();
        }

        fn transfer_to(&self, _my_memory: &Any, _src_device: &Any, _src_memory: &Any)
                    -> Result<(), Error> {
            unimplemented!();
        }
    }


    #[test]
    fn it_works() {
        let mut shared = SharedTensor::new(vec![1,2,3]);

        let dev = CpuDevice {index: 0};
        // let t1 = shared.read(&dev);
        // let t2 = shared.read(&dev);
        {
            let tmut = shared.write_only(&dev);
        }

        let t2 = shared.read(&dev).unwrap();
        // let tmut2 = shared.write_only(&dev);
    }
}
