use std::pin::Pin;

use async_trait::async_trait;
use futures::{stream, Stream};

use crate::{
    document_loaders::{process_doc_stream, Loader, LoaderError},
    schemas::Document,
    text_splitter::TextSplitter,
};

#[derive(Debug, Clone)]
pub struct TextLoader {
    content: String,
}

impl TextLoader {
    pub fn new<T: Into<String>>(input: T) -> Self {
        Self {
            content: input.into(),
        }
    }
}

#[async_trait]
impl Loader for TextLoader {
    async fn load(
        mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let doc = Document::new(self.content);
        let stream = stream::iter(vec![Ok(doc)]);
        Ok(Box::pin(stream))
    }

    async fn load_and_split<TS: TextSplitter + 'static>(
        mut self,
        splitter: TS,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let doc_stream = self.load().await?;
        let stream = process_doc_stream(doc_stream, splitter);
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;

    use crate::text_splitter::TokenSplitter;

    use super::*;

    #[tokio::test]
    async fn test_reading_mocked_file_content() {
        let mocked_file_content = r#"
iterary Descriptive Text
Doña Uzeada de Ribera Maldonado de Bracamonte y Anaya was short, plump, and mustachioed. There was no longer any reason to call hers a figure. Her vibrant, healthy colors could overcome the lead white and ceruse she used for makeup to feign melancholies. She wore two dark patches adhered to her temples, pretending to be medicines. She had small, mischievous, mouse-like eyes. She knew how to dilate them sternly, dim them modestly, or raise them subtly. She walked swaying her impossible hips, and it was difficult, upon seeing her, not to associate her squat image with that of certain domestic waterfowl. Blue and azure rings choked her phalanges.
	•	Manuel Mujica Lainez, Don Galaz de Buenos Aires
The descriptive text, in this case, a portrait of a person, evokes such an image in the receiver that the described reality takes shape, materializes in their mind. In this case, the text talks about a real character: Doña Uzeada de Ribera Maldonado de Bracamonte y Anaya. As it is a literary description, the attitude of the emitter is subjective, as it aims to transmit their own personal vision in the description, and the language function is predominantly poetic, as it seeks a particular aesthetic.
Non-Literary Descriptive Text
The west of Texas divides the border between Mexico and New Mexico. It is very beautiful but rugged, filled with cacti; in this region are found the Davis Mountains. The entire terrain is filled with limestone, twisted mesquite trees, and prickly pear cactuses. To admire the true desert beauty, visit Big Bend National Park, near Brownsville. It is a favorite location for hikers, campers, and rock enthusiasts. Small towns and ranches lie along the plains and canyons of this region. The area only has two seasons, mild and really hot. The best time to visit is from December to March when the days are warm, the nights are cool, and the desert plants bloom with moisture in the air.

"#;

        // Create a new TextLoader with the mocked content
        let loader = TextLoader::new(mocked_file_content.to_string());

        // Use the loader to load the content, which should be wrapped in a Document
        let mut documents = loader.load().await.unwrap();
        while let Some(doc) = documents.next().await {
            assert_eq!(doc.unwrap().page_content, mocked_file_content); // Ensure the Document contains the mocked content
        }

        let loader = TextLoader::new(mocked_file_content.to_string());
        let splitter = TokenSplitter::default();

        let mut documents = loader.load_and_split(splitter).await.unwrap();

        while let Some(doc) = documents.next().await {
            println!("{:?}", doc.unwrap());
            println!("/n");
            println!("-----------------------");
        }
    }
}
