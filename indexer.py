# indexer.py
import os, uuid
from typing import List, Dict
from langchain_community.vectorstores import Weaviate as LCWeaviate
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Embeddings: either SentenceTransformers (no network) or OpenAI
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
try:
    if EMBEDDING_MODEL.startswith("text-embedding"):
        from langchain_openai import OpenAIEmbeddings
        Emb = lambda: OpenAIEmbeddings(model=EMBEDDING_MODEL)
    else:
        from langchain.embeddings import HuggingFaceEmbeddings
        Emb = lambda: HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
except Exception as e:
    raise RuntimeError(f"Embeddings backend not available: {e}")

def to_docs(invoices: List[Dict], company_slug: str) -> List[Document]:
    docs = []
    for inv in invoices:
        txt = []
        # collect whatever fields your tenant returns
        issue_date = inv.get("issueDate") or inv.get("issuedDate") or ""
        inv_no = inv.get("invoiceNumber") or inv.get("number") or ""
        supplier = (inv.get("customer") or {}).get("name") or ""
        total = inv.get("total") or inv.get("amount") or None
        vat = inv.get("vatTotal") or None
        currency = inv.get("currency") or "NOK"

        # main text (you can expand with line-items, notes, etc.)
        txt.append(f"Invoice {inv_no} dated {issue_date} for {supplier}")
        if total:
            txt.append(f"Total {total} {currency}")
        if vat:
            txt.append(f"VAT {vat} {currency}")
        if inv.get("comment"):
            txt.append(f"Comment: {inv.get('comment')}")

        # If you have OCR results for attachments, append here:
        # txt.append("OCR: ...")

        docs.append(Document(
            page_content="\n".join([t for t in txt if t]),
            metadata={
                "company_slug": company_slug,
                "doc_type": "invoice",
                "invoice_no": inv_no,
                "issue_date": issue_date,
                "supplier": supplier,
                "currency": currency,
                "amount_total": float(total) if total is not None else None,
                "vat_total": float(vat) if vat is not None else None,
                # "source_url": signed download link if you have one
            }
        ))
    return docs

def upsert_invoices_weaviate(client, invoices: List[Dict], company_slug: str):
    # LangChain’s Weaviate wrapper uses client by URL/api_key, so we pass params:
    url = os.environ["WEAVIATE_URL"]
    api_key = os.environ["WEAVIATE_API_KEY"]

    docs = to_docs(invoices, company_slug)
    if not docs:
        return 0

    _ = LCWeaviate.from_documents(
        documents=docs,
        embedding=Emb(),
        weaviate_url=url,
        index_name="InvoiceChunk",
        text_key="text",                      # property that will store page_content
        by_text=False,                        # we push our own vectors
        weaviate_api_key=api_key,
        metadata_keys=[
            "company_slug","doc_type","invoice_no","issue_date",
            "supplier","currency","amount_total","vat_total","source_url"
        ],
    )
    return len(docs)
