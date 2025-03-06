use std::collections::HashSet;

use crate::chain::Chain;

use super::SequentialChain;

pub struct SequentialChainBuilder {
    chains: Vec<Box<dyn Chain>>,
}

impl SequentialChainBuilder {
    pub fn new() -> Self {
        Self { chains: Vec::new() }
    }

    pub fn add_chain<C: Chain + 'static>(mut self, chain: C) -> Self {
        self.chains.push(Box::new(chain));
        self
    }

    pub fn build(self) -> SequentialChain {
        let outputs: HashSet<String> = self
            .chains
            .iter()
            .flat_map(|c| c.get_output_keys())
            .collect();

        let input_keys: HashSet<String> = self
            .chains
            .iter()
            .flat_map(|c| c.get_input_keys())
            .collect();

        SequentialChain {
            chains: self.chains,
            input_keys,
            outputs,
        }
    }
}

impl Default for SequentialChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[macro_export]
macro_rules! sequential_chain {
    ( $( $chain:expr ),* $(,)? ) => {
        {
            let mut builder = $crate::chain::SequentialChainBuilder::new();
            $(
                builder = builder.add_chain($chain);
            )*
            builder.build()
        }
    };
}
