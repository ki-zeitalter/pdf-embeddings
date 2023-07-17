# pdf-embeddings
Eine Demo-Web-Anwendung, die die Verwendung von Embeddings zeigt. Die Anwendung erstellt aus deinen Dokumenten Embeddings und ermöglicht es dir, über einen Suchtext, die relevanten Textstellen zu extrahieren.
Das ist u.a. dafür nützlich, wenn man große bzw. viele PDF-Dateien hat und diese über eine KI auswertbar machen möchte. Zu langer Text würde das Token-Limit der KI sprengen und mithilfe von Embeddings, können wir 
vorher schon die relevanten Textstellen ermitteln.

## Vorbereitungen

Erst musst du die notwendigen Bibliotheken installieren:

`pip install langchain openai python-dotenv faiss-cpu pypdf2 streamlit tiktoken`

Erstele eine Kopie von *.env.template* als *.env* und füge deinen OpenAI API Key ein. Die *.env* Datei wird nicht in Git eingecheckt.

## Starten der Anwendung
Da wir Streamlit nutzen, können wir nicht direkt den Python-Befehl nutzen. Statt dessen musst du folgendes Kommando ausführen:

`streamlit run embeddings.py`