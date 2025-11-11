# vec.py
import os
import weaviate
from weaviate.classes.config import Property, DataType, Configure

def get_client():
    url = os.environ["WEAVIATE_URL"]
    key = os.environ["WEAVIATE_API_KEY"]
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=weaviate.auth.AuthApiKey(key),
        # turn off if your WCS project already has server-side vectorizers:
        headers={}
    )

def ensure_schema(client):
    # One class for TEXT chunks from invoices/ocr
    if not client.collections.exists("InvoiceChunk"):
        client.collections.create(
            name="InvoiceChunk",
            vectorizer_config=Configure.Vectorizer.none(),  # we'll push our own vectors from LangChain
            properties=[
                Property(name="company_slug", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="invoice_no", data_type=DataType.TEXT),
                Property(name="issue_date", data_type=DataType.TEXT),
                Property(name="supplier", data_type=DataType.TEXT),
                Property(name="currency", data_type=DataType.TEXT),
                Property(name="amount_total", data_type=DataType.NUMBER),
                Property(name="vat_total", data_type=DataType.NUMBER),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source_url", data_type=DataType.TEXT),  # if you have a signed link
            ],
        )

    # Optional: separate class for IMAGE vectors from receipts
    if not client.collections.exists("AttachmentImage"):
        client.collections.create(
            name="AttachmentImage",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="company_slug", data_type=DataType.TEXT),
                Property(name="invoice_no", data_type=DataType.TEXT),
                Property(name="issue_date", data_type=DataType.TEXT),
                Property(name="supplier", data_type=DataType.TEXT),
                Property(name="caption", data_type=DataType.TEXT),  # OCR summary
                Property(name="source_url", data_type=DataType.TEXT),
            ],
        )
