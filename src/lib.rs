use std::borrow::{Cow, Borrow};
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;
use std::ops::Deref;

pub trait Framework {
    type D: Clone + Eq + Any;
    type E;
    type M: Memory + Any;

    fn allocate_memory(&Self::D, usize) -> Result<Self::M, Error>;
}

trait MemoryTransfer {
    type F: Framework;

    fn can_transfer_to<FD: Framework>() -> bool;
    fn transfer_to<FD: Framework>(&Self, memory: F::M) -> Result<(), Error>;
}

// trait Device {
//     type M;
// }


pub struct Tensor<'a, F: Framework> {
    dim: Cow<'a, [usize]>,
    memory: Rc<F::M>,
}

pub enum Error {
    UninitializedMemory,
    AllocationFailed,
}

type Version = u32;

struct TensorLocation {
    device: Box<Any>,
    version: Version,
    mem: Rc<Any>,
}


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

    pub fn size(self) -> usize {
        1 // FIXME
    }

    /// Looks up `device` in self.locations and returns its index. If lookup
    /// fails then new location is created.
    fn get_location_index<F: Framework>(&self, device: &F::D)
                                        -> Result<usize, Error> {
        for (i, loc) in self.locations.borrow().iter().enumerate() {
            match loc.device.deref().downcast_ref::<&F::D>() {
                Some(ref d) if **d == device => return Ok(i),
                _ => {},
            }
        }

        self.locations.borrow_mut().push(TensorLocation {
            device: Box::new(device.clone()),
            version: 0,
            mem: Rc::new(try!(F::allocate_memory(device, self.size()))),
        });
        Ok(self.locations.borrow().len() - 1)
    }

    /// TODO: chose the best source to copy data from.
    /// That would require some additional traits that return costs for
    /// transferring data between different backends.
    fn update_if_needed<F: Framework>(&self, device: &F::D, dest_i: usize)
                                      -> Result<(), Error> {
        let locs = self.locations.borrow_mut();
        if locs[dest_i].version == self.latest_version {
            return Ok(());
        }

        let src_i = locs.iter().enumerate()
            .filter(|&(i, ref loc)| loc.version == self.latest_version)
            .next().expect("broken invariant: can't find latest version");

    }



    pub fn read<'a, F: Framework>(&'a self, device: &F::D)
                                  -> Result<Tensor<'a, F>, Error> {
        if self.latest_version == 0 {
            return Err(Error::UninitializedMemory);
        }
        let i = try!(self.get_location_index(device));
        try!(self.update_if_needed(device, i));

        Ok(Tensor {
            dim: Cow::Borrowed(&self.dim)
        })
    }

    // pub fn read_write<'a>(&'a mut self, device: &Device) -> Result<Tensor<'a>, Error> {
    //     Ok(Tensor {dim: Cow::Borrowed(&self.dim)})
    // }

    // pub fn write_only<'a>(&'a mut self, device: &Device) -> Result<Tensor<'a>, Error> {
    //     Ok(Tensor {dim: Cow::Borrowed(&self.dim)})
    // }
}


// #[cfg(test)]
// mod tests {
//     use super::*;
//     #[test]
//     fn it_works() {
//         let dev = Device {name: "cuda".to_string()};

//         let mut shared = SharedTensor::new(vec![1,2,3]);
//         let t1 = shared.read(&dev);
//         // let tmut = shared.write_only(&dev);
//     }
// }
