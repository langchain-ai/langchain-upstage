# langchain-upstage

This package contains the LangChain integrations for [Upstage](https://upstage.ai) through their [APIs](https://developers.upstage.ai/docs/getting-started/models).

## Installation and Setup

- Install the LangChain partner package
```bash
pip install -U langchain-upstage
```

- Get an Upstage api key from [Upstage Console](https://console.upstage.ai/home) and set it as an environment variable (`UPSTAGE_API_KEY`)

## Chat Models

This package contains the `ChatUpstage` class, which is the recommended way to interface with Upstage models.

See a [usage example](https://python.langchain.com/docs/integrations/chat/upstage)

## Embeddings

See a [usage example](https://python.langchain.com/docs/integrations/text_embedding/upstage)

Use `solar-embedding-1-large` model for embeddings. Do not add suffixes such as `-query` or `-passage` to the model name.
`UpstageEmbeddings` will automatically add the suffixes based on the method called.

## Document Parse Loader

See a [usage example](https://python.langchain.com/v0.1/docs/integrations/document_loaders/upstage/)

The `use_ocr` option determines whether OCR will be used for text extraction from documents. If this option is not specified, the default policy of the [Upstage Document Parse API](https://console.upstage.ai/docs/capabilities/document-parse#request) service will be applied. When `use_ocr` is set to `True`, OCR is utilized to extract text. In the case of PDF documents, this involves converting the PDF into images before performing OCR. Conversely, if `use_ocr` is set to `False` for PDF documents, the text information embedded within the PDF is used directly. However, if the input document is not a PDF, such as an image, setting `use_ocr` to `False` will result in an error.

```python
from langchain_upstage import UpstageDocumentParseLoader

file_path = "/PATH/TO/YOUR/FILE.image"
layzer = UpstageDocumentParseLoader(file_path, split="page")

# For improved memory efficiency, consider using the lazy_load method to load documents page by page.
docs = layzer.load()  # or layzer.lazy_load()

for doc in docs[:3]:
    print(doc)
```

If you are a Windows user, please ensure that the [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) is installed before using the loader.
