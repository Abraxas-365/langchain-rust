use std::collections::HashSet;

use crate::{chain::Chain, prompt::PromptArgs};

use super::SequentialChain;

pub struct SequentialChainBuilder<T>
where
    T: PromptArgs,
{
    chains: Vec<Box<dyn Chain<T>>>,
}

impl<T> SequentialChainBuilder<T>
where
    T: PromptArgs,
{
    pub fn new() -> Self {
        Self { chains: Vec::new() }
    }

    pub fn add_chain<C: Chain<T> + 'static>(mut self, chain: C) -> Self {
        self.chains.push(Box::new(chain));
        self
    }

    pub fn build(self) -> SequentialChain<T> {
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

impl<T> Default for SequentialChainBuilder<T>
where
    T: PromptArgs,
{
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
