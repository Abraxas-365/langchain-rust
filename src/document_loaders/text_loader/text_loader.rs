use std::pin::Pin;

use async_trait::async_trait;
use futures::{stream, Stream, StreamExt};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::{
    document_loaders::{Loader, LoaderError},
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
    ) -> Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + Sync + 'static>> {
        let doc = Document::new(self.content);
        let stream = stream::iter(vec![Ok(doc)]);
        Box::pin(stream)
    }

    async fn load_and_split<TS: TextSplitter + 'static>(
        mut self,
        splitter: TS,
    ) -> Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + Sync + 'static>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut doc_stream = self.load().await;
            while let Some(doc) = doc_stream.next().await {
                if let Ok(doc) = doc {
                    let doc_splitted = splitter.split_documents(&[doc]);
                    match doc_splitted {
                        Ok(docs) => {
                            for doc in docs {
                                let _ = tx.send(Ok(doc));
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(LoaderError::TextSplitterError(e)));
                        }
                    }
                }
            }
        });
        let stream = UnboundedReceiverStream::new(rx);
        Box::pin(stream)
    }
}

#[cfg(test)]
mod tests {
    use crate::text_splitter::TokenSplitter;

    use super::*;

    #[tokio::test]
    async fn test_reading_mocked_file_content() {
        let mocked_file_content = r#"
Doña Uzeada de Ribera Maldonado de Bracamonte y Anaya era baja, rechoncha, abigotada. Ya no existia razon para llamar talle al suyo. Sus colores vivos, sanos, podian mas que el albayalde y el soliman del afeite, con que se blanqueaba por simular melancolias. Gastaba dos parches oscuros, adheridos a las sienes y que fingian medicamentos. Tenia los ojitos ratoniles, maliciosos. Sabia dilatarlos duramente o desmayarlos con recato o levantarlos con disimulo. Caminaba contoneando las imposibles caderas y era dificil, al verla, no asociar su estampa achaparrada con la de ciertos palmipedos domesticos. Sortijas celestes y azules le ahorcaban las falanges

Manuel Mujica Lainez, Don Galaz de Buenos Aires

El texto descriptivo, en este caso un retrato de una persona, provoca en el receptor una imagen tal que la realidad descripta cobra forma, se materializa en su mente. En este caso el texto habla de un personaje real: Doña Uzeada de Ribera Maldonado de Bracamonte y Anaya. Como se trata de una descripcion literaria, la actitud del emisor es subjetiva, dado que pretende transmitir su propia vision personal al describir y la funcion del lenguaje es predominantemente poetica, ya que persigue una estetica en particular.

Ejemplo de texto descriptivo no literario
El oeste de Texas divide la frontera entre Mexico y Nuevo México. Es muy bella pero aspera, llena de cactus, en esta region se encuentran las Davis Mountains. Todo el terreno esta lleno de piedra caliza, torcidos arboles de mezquite y espinosos nopales. Para admirar la verdadera belleza desertica, visite el Parque Nacional de Big Bend, cerca de Brownsville. Es el lugar favorito para los excurcionistas, acampadores y entusiastas de las rocas. Pequeños pueblos y ranchos se encuentran a lo largo de las planicies y cañones de esta region. El area solo tiene dos estaciones, tibia y realmente caliente. La mejor epoca para visitarla es de Diciembre a Marzo cuando los dias son tibios, las noches son frescas y florecen las plantas del desierto con la humedad en el aire.

"#;

        // Create a new TextLoader with the mocked content
        let loader = TextLoader::new(mocked_file_content.to_string());

        // Use the loader to load the content, which should be wrapped in a Document
        let mut documents = loader.load().await;
        while let Some(doc) = documents.next().await {
            assert_eq!(doc.unwrap().page_content, mocked_file_content); // Ensure the Document contains the mocked content
        }

        let loader = TextLoader::new(mocked_file_content.to_string());
        let splitter = TokenSplitter::default();

        let mut documents = loader.load_and_split(splitter).await;

        while let Some(doc) = documents.next().await {
            println!("{:?}", doc.unwrap());
            println!("/n");
            println!("-----------------------");
        }
    }
}
