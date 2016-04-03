#![feature(get_type_id)]
/// This module is an attempt to figure out how to improve structure of
/// `collenchyma`.
///
/// Complete:
/// - Allow to separate CUDA, OpenCL and Native backends into their own
///   independent crates. No need to specify features during compilation,
///   just import `collenchyma-cuda`, `collenchyma-opencl` as required.
///   This doesn't come for free, though.
/// - Keep currently implemented restriction on memory mutability: allow only
///   one mutable "view" of `SharedTensor` (implemeted as `MutTensor`), or many
///   immutable views (`Tensor`). Since all comutations are done only on `Tensor`
///   and `MutTensor` there should be no data races.
/// - Don't do synchronization when it's not strictly required. E.g. if copies
///   on both GPU and CPU are current, getting read references to any of them
///   won't result in redundant transfers.
/// - Implement lightweight protection from use of unitialized memory in
///   runtime.
/// - Skip initialization for write-only views.
/// - Allow syncs and mem allocations on immutable `SharedTensor`. Now it needs
///   to be mutable only for shape and data mutation and data invalidation.
/// - `Tensor` and `MutTensor` are parametrized on backend driver and give out
///   typed memory on request, so there is no need to use .as_native().unwrap(),
///   etc. If device was Native, mem has to be Native also. Additionally it also
///   makes impossilbe to pass Native memory to Cuda backend operations.
/// - Both `Tensor` and `MutTensor` can be reshaped in-place. Reshaping doesn't
///   affect their parent `SharedTensor`. There is quite a lot of reshaping in
///   Leaf code, so it's nice to have special support for it.
///
/// Possible:
/// - I'd suggest implementing everything related to dimensions directly on
///   `Tensor`, `MutTensor` and `SharedTensor`. It would shorten
///   `x.desc().dim()` to `x.dim()`. The calls are quite common and I think
///   it's worth doing. More importantly it also removes `TensorDesc` entity
///   and reduces mental load.
/// - It shouldn't be difficult to implement slicing, but that would probably
///   require implementing slicing for memory first. Slicing can be used to cut
///   number of memory transfers. E.g. solver may need to update scalars for
///   local learning rate on GPU for each of `N` weight tensors every epoch.
///   That'll result in `N` 4 byte transfers from host to GPU that could be
///   replaced with single transfer with slicing.
/// - Async operations: it looks like currently most time is spent waiting
///   for in/out transfers even on mid-range GPU hardware. Async may help a lot.
///   Async can be implemented by making transfer_in/transfer_out to return
///   an object that can be waited on until transfer completes when sync
///   is required, e.g. CUDA -> Host. `Tensor::get_memory()` could block
///   until transfer completes.
/// - Currently code assumes that cloning `Device` is cheap. It's possible
///   to replace copy with a pointer in `Tensor` and `MutTensor`, if it's a
///   problem.
///
/// Not implemented:
/// - while currently it's not possible to mismatch tensors with different
///   device types, it's still possible to cause mismatch with different
///   devices of one type. Can something be done about it?
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
/// - Code in Leaf reshapes input tensors every forward pass. It's possible
///   to share memory allocations between several `SharedTensor`s and allow
///   each one have its own shape. But that would complicate internals a lot.
///   Data race policies (mut/immut borrows) would have to be checked in
///   runtime (currently they are checked at compile time). It would also
///   complicate mental model a lot since several tensors would magically share
///   the same content and this destroys locality.

use std::any::Any;
use std::borrow::{Cow, Borrow};
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Deref;

pub mod native;

/// This trait should be implemented for `Device`.
/// Use of `Any` everywhere is ugly, but it looks like there is no other way
/// to do it if we want to extract CUDA stuff into its own crate completely,
/// so that base crate knows nothing about it at all.
pub trait MemoryTransfer {
    fn transfer_in(&self, my_memory: &mut Any, src_device: &Any, src_memory: &Any)
                   -> Result<(), Error>;
    fn transfer_out(&self, my_memory: &Any, dst_device: &Any, dst_memory: &mut Any)
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
/// Data referred by this tensor is immutable while shape is mutable.
/// TODO: define methods like reshape(), get_mem(), etc.
pub struct Tensor<'a, D: Device> {
    dim: Cow<'a, [usize]>,
    device: D,
    memory: &'a D::M,
}

impl <'a, D: Device> Tensor<'a, D> {
    pub fn size(&self) -> usize {
        self.dim.iter().fold(1, |s, &x| s * x)
    }

    pub fn device(&self) -> &D {
        &self.device
    }

    pub fn mem(&'a self) -> &'a D::M {
        self.memory
    }
}

/// `MutTensor` located on a specific device.
/// Data referred by this tensor and its shape are mutable.
/// TODO: define methods like reshape(), get_mem(), .get_mut_mem(), .fill(), etc.
pub struct MutTensor<'a, D: Device> {
    dim: Cow<'a, [usize]>,
    device: D,
    memory: &'a mut D::M,
}

impl <'a, D: Device> MutTensor<'a, D> {
    pub fn size(&self) -> usize {
        self.dim.iter().fold(1, |s, &x| s * x)
    }

    pub fn device(&self) -> &D {
        &self.device
    }

    pub fn mut_mem(&'a mut self) -> &'a mut D::M {
        self.memory
    }
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
    mem: Box<Any>,
}

/// `SharedTensor` keeps track of all memory locations and their versions
/// and does memory transfers when they are required.
/// TODO: impl resize(), reshape(), etc.
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
            }
        }

        let dev_clone: D = device.clone();
        {
            let dev_any: Box<Any> = Box::new(dev_clone);
            println!("Device={:?} typeid={:?}", dev_any, dev_any.get_type_id());
        }

        self.locations.borrow_mut().push(TensorLocation {
            device: Box::new(device.clone()),
            version: 0,
            mem: Box::new(try!(D::allocate_memory(device, self.size()))),
        });
        Ok(self.locations.borrow().len() - 1)
    }

    /// TODO: chose the best source to copy data from.
    /// That would require some additional traits that return costs for
    /// transferring data between different backends.
    /// Actually I think that there would be only transfers between
    /// `Native` <-> `Cuda` and `Native` <-> `OpenCL` in foreseeable future,
    /// so it's best to not overengineer here.
    fn update_if_needed(&self, dst_i: usize)
                                   -> Result<(), Error> {
        let mut locs = self.locations.borrow_mut();
        if locs[dst_i].version == self.latest_version {
            return Ok(());
        }

        // for (i, loc) in locs.iter().enumerate() {
        //     println!("i={} ver={} latest={}", i, loc.version, self.latest_version);
        // }

        // TODO: filter out impossible transfers?
        let (src_i, _) = locs.iter().enumerate()
            .filter(|&(_, loc)| loc.version == self.latest_version)
            .next().expect("broken invariant: can't find latest version");

        // We need to borrow two different Vec elements: src and mut dst.
        // Borrowck doesn't allow to do it in a straightforward way, so
        // here is the shortest method I could think of.
        assert!(src_i != dst_i);
        let (src_loc, mut dst_loc) = if src_i < dst_i {
            let (left, right) = locs.split_at_mut(dst_i);
            (&left[src_i], &mut right[0])
        } else {
            let (left, right) = locs.split_at_mut(src_i);
            (&right[0], &mut left[dst_i])
        };

        // Backends may define transfers asymmetrically. E. g. CUDA may know how
        // to transfer to and from Native backend, while Native may know nothing
        // about CUDA at all. So we try to change device that does transfer
        // if first attempt fails.
        let src_device = src_loc.device.deref().downcast_ref::<native::NativeDevice>()
            .expect("broken invariant: object doesn't support mem transfers");
        match src_device.transfer_out(src_loc.mem.deref(),
                                      dst_loc.device.deref(), dst_loc.mem.as_mut()) {
            Err(Error::NoMemoryTransferRoute) => {},
            x => return x,
        }

        let dst_device = dst_loc.device.deref().downcast_ref::<native::NativeDevice>()
            .expect("broken invariant: object doesn't support mem transfers");
        match dst_device.transfer_in(dst_loc.mem.as_mut(),
                                     src_loc.device.deref(), src_loc.mem.deref()) {
            Err(Error::NoMemoryTransferRoute) => {},
            x => return x,
        }
        // TODO: fallback on indirect transfer via Native
        Err(Error::NoMemoryTransferRoute)
    }

    /// Get tensor for reading on the specified `device`.
    /// Can fail if memory allocation fails, or tensor wasn't initialized yet.
    pub fn read<'a, D: Device>(&'a self, device: &D)
                               -> Result<Tensor<'a, D>, Error> {
        if self.latest_version == 0 {
            return Err(Error::UninitializedMemory);
        }
        let i = try!(self.get_location_index::<D>(device));
        try!(self.update_if_needed(i));

        let locs = self.locations.borrow();
        let mem: &D::M = locs[i].mem.deref().downcast_ref::<D::M>()
                            .expect("Broken invariant: wrong memory type");
        let mem_a: &'a D::M = unsafe { ::std::mem::transmute(mem) };
        Ok(Tensor {
            dim: Cow::Borrowed(&self.dim),
            memory: mem_a,
            device: device.clone(),
        })
    }

    /// Get tensor for reading and writing on the specified `device`.
    /// This memory location is set as latest.
    /// Can fail if memory allocation fails, or tensor wasn't initialized yet.
    pub fn read_write<'a, D: Device>(&'a self, _device: &D)
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
        locs[i].version = self.latest_version;

        let mem: &mut D::M = locs[i].mem.as_mut().downcast_mut::<D::M>()
                            .expect("Broken invariant: wrong memory type");
        let mem_a: &'a mut D::M = unsafe { ::std::mem::transmute(mem) };
        Ok(MutTensor {
            dim: Cow::Borrowed(&self.dim),
            memory: mem_a,
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
    use super::*;
    use native::NativeDevice;

    #[test]
    fn it_works() {
        let mut shared = SharedTensor::new(vec![1,2,3]);

        let dev = NativeDevice::new(0);
        let dev2 = NativeDevice::new(1);
        let dev3 = NativeDevice::new(2);

        // let t0 = shared.read(&dev);

        {
            let mut t1 = shared.write_only(&dev).unwrap();
            for x in t1.mut_mem().as_mut_slice() {
                *x = 11;
            }

        }

        {
            for x in shared.write_only(&dev).unwrap().mut_mem().as_mut_slice() {
                *x = 11;
            }
        }


        // let mem = {
        {
            let t2 = shared.read(&dev).unwrap();
            let _t3 = shared.read(&dev).unwrap();
            let t2_mem = t2.mem();
            println!("mem {:?}", t2_mem);
            println!("mem {:?}", shared.read(&dev).unwrap().mem());
            // t2_mem
            // println!("mem {:?}", shared.read(&dev).unwrap().mem());
        }

        {
            let _tmut = shared.write_only(&dev);
        }

        // shared.write_only(&dev2);
        println!("mem at dev2 {:?}", shared.read(&dev2).unwrap().mem());
        println!("mem at dev3 {:?}", shared.read(&dev3).unwrap().mem());
        // let tmut2 = shared.write_only(&dev);
        assert!(false);
    }
}
