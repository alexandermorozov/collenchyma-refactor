use std::any::Any;
use {Error, MemoryTransfer, Device};


#[derive(Debug)]
pub struct NativeMemory {
    data: Vec<u8>
}

impl NativeMemory {
    // Toy impl just to test that it works

    /// Size in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }
    
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
}


#[derive(PartialEq, Eq, Clone, Debug)]
pub struct NativeDevice {
    numa_cluster: usize,
}

impl NativeDevice {
    pub fn new(numa_cluster: usize) -> NativeDevice {
        NativeDevice {numa_cluster: numa_cluster}
    }
}

impl Device for NativeDevice {
    type M = NativeMemory;

    fn allocate_memory(_dev: &Self, size: usize) -> Result<Self::M, Error> {
        Ok(NativeMemory {
            data: vec![0; size]
        })
    }
}

impl MemoryTransfer for NativeDevice {
    fn transfer_out(&self, my_memory: &Any, dst_device: &Any, dst_memory: &mut Any)
                    -> Result<(), Error> {
        if let Some(_) = dst_device.downcast_ref::<NativeDevice>() {
            let my_mem = my_memory.downcast_ref::<NativeMemory>().unwrap();
            let mut dst_mem = dst_memory.downcast_mut::<NativeMemory>().unwrap();
            dst_mem.as_mut_slice().clone_from_slice(my_mem.as_slice());
            return Ok(());
        }

        Err(Error::NoMemoryTransferRoute)
    }

    fn transfer_in(&self, my_memory: &mut Any, src_device: &Any, src_memory: &Any)
                   -> Result<(), Error> {
        if let Some(_) = src_device.downcast_ref::<NativeDevice>() {
            let mut my_mem = my_memory.downcast_mut::<NativeMemory>().unwrap();
            let src_mem = src_memory.downcast_ref::<NativeMemory>().unwrap();
            my_mem.as_mut_slice().clone_from_slice(src_mem.as_slice());
            return Ok(());
        }

        Err(Error::NoMemoryTransferRoute)
    }
}
