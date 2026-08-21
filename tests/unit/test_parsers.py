"""Tests for inference_server/parsers.py."""

from __future__ import annotations

import json

import pytest

# ── Module-level constants ──────────────────────────────────────────────
SAMPLE_CSV = "name,age\nAlice,30\nBob,25"
SAMPLE_JSON = '{"key": "value", "num": 42}'
SAMPLE_JSONL = '{"role": "user", "content": "hi"}\n{"role": "assistant", "content": "hello"}'
SAMPLE_XML = '<root><item id="1">Hello</item><item id="2">World</item></root>'
SAMPLE_HTML = "<html><head><title>T</title></head><body><p>Hello</p><script>x=1</script><footer>Footer</footer></body></html>"
SAMPLE_RTF = r"{\rtf1\ansi Hello \b World\b0}"
SAMPLE_ODT_ZIP_CONTENT = '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text"><office:body><office:text><text:p>Hello from ODT</text:p></office:text></office:body></office:document-content>'


@pytest.mark.unit
class TestParseText:
    def test_reads_text_file(self, sample_text_file):
        from inference_server.parsers import parse_text
        result = parse_text(sample_text_file)
        assert "Hello world" in result

    def test_returns_empty_for_empty_file(self, tmp_dir):
        from inference_server.parsers import parse_text
        p = tmp_dir / "empty.txt"
        p.write_text("")
        assert parse_text(p) == ""


@pytest.mark.unit
class TestParseCsv:
    def test_csv_to_markdown_table(self, tmp_dir):
        from inference_server.parsers import parse_csv
        p = tmp_dir / "data.csv"
        p.write_text(SAMPLE_CSV)
        result = parse_csv(p)
        assert "name | age" in result
        assert "---" in result
        assert "Alice | 30" in result

    def test_empty_csv(self, tmp_dir):
        from inference_server.parsers import parse_csv
        p = tmp_dir / "empty.csv"
        p.write_text("")
        assert parse_csv(p) == ""


@pytest.mark.unit
class TestParseJson:
    def test_valid_json(self, tmp_dir):
        from inference_server.parsers import parse_json
        p = tmp_dir / "data.json"
        p.write_text(SAMPLE_JSON)
        result = parse_json(p)
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["num"] == 42

    def test_invalid_json_raises(self, tmp_dir):
        from inference_server.parsers import parse_json
        p = tmp_dir / "bad.json"
        p.write_text("{not json}")
        with pytest.raises(json.JSONDecodeError):
            parse_json(p)


@pytest.mark.unit
class TestParseJsonl:
    def test_valid_jsonl(self, tmp_dir):
        from inference_server.parsers import parse_jsonl
        p = tmp_dir / "data.jsonl"
        p.write_text(SAMPLE_JSONL)
        result = parse_jsonl(p)
        assert "user" in result
        assert "assistant" in result

    def test_malformed_line_preserved(self, tmp_dir):
        from inference_server.parsers import parse_jsonl
        p = tmp_dir / "mixed.jsonl"
        p.write_text('{"ok": true}\nnot json\n{"also": true}')
        result = parse_jsonl(p)
        assert "not json" in result
        assert "ok" in result

    def test_empty_jsonl(self, tmp_dir):
        from inference_server.parsers import parse_jsonl
        p = tmp_dir / "empty.jsonl"
        p.write_text("")
        assert parse_jsonl(p) == ""


@pytest.mark.unit
class TestParseXml:
    def test_valid_xml(self, tmp_dir):
        from inference_server.parsers import parse_xml
        p = tmp_dir / "data.xml"
        p.write_text(SAMPLE_XML)
        result = parse_xml(p)
        assert "Hello" in result
        assert "World" in result

    def test_invalid_xml_falls_back_to_text(self, tmp_dir):
        from inference_server.parsers import parse_xml
        p = tmp_dir / "bad.xml"
        p.write_text("<<not xml>>")
        result = parse_xml(p)
        assert "<<not xml>>" in result


@pytest.mark.unit
class TestParseHtml:
    def test_html_extracts_text(self, tmp_dir):
        from inference_server.parsers import parse_html
        p = tmp_dir / "page.html"
        p.write_text(SAMPLE_HTML)
        result = parse_html(p)
        assert "Hello" in result
        # script/footer should be stripped
        assert "x=1" not in result

    def test_html_fallback_without_bs4(self, tmp_dir, mocker):
        from inference_server.parsers import parse_html
        # Force ImportError for bs4
        real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
        def fake_import(name, *args, **kwargs):
            if name == "bs4":
                raise ImportError("no bs4")
            return real_import(name, *args, **kwargs)
        mocker.patch("builtins.__import__", side_effect=fake_import)
        p = tmp_dir / "page2.html"
        p.write_text(SAMPLE_HTML)
        result = parse_html(p)
        assert "Hello" in result


@pytest.mark.unit
class TestParsePdf:
    def test_pdf_fallback_message(self, mocker, tmp_dir):
        """When no PDF library is available, should return fallback string."""
        from inference_server.parsers import parse_pdf
        p = tmp_dir / "doc.pdf"
        p.write_bytes(b"%PDF-1.4 fake")
        # Patch all PDF libraries to be unavailable
        mocker.patch.dict("sys.modules", {
            "pypdf": None,
            "PyPDF2": None,
        }, clear=False)
        mocker.patch("subprocess.run", side_effect=FileNotFoundError)
        result = parse_pdf(p)
        assert "pypdf" in result.lower() or "install" in result.lower()


@pytest.mark.unit
class TestParseDocx:
    def test_docx_fallback(self, mocker, tmp_dir):
        from inference_server.parsers import parse_docx
        p = tmp_dir / "doc.docx"
        p.write_bytes(b"fake docx")
        mocker.patch.dict("sys.modules", {"docx": None}, clear=False)
        result = parse_docx(p)
        assert "python-docx" in result.lower() or "docx" in result.lower()


@pytest.mark.unit
class TestParseXlsx:
    def test_xlsx_fallback(self, mocker, tmp_dir):
        from inference_server.parsers import parse_xlsx
        p = tmp_dir / "data.xlsx"
        p.write_bytes(b"fake xlsx")
        mocker.patch.dict("sys.modules", {"openpyxl": None}, clear=False)
        result = parse_xlsx(p)
        assert "openpyxl" in result.lower() or "xlsx" in result.lower()


@pytest.mark.unit
class TestParsePptx:
    def test_pptx_fallback(self, mocker, tmp_dir):
        from inference_server.parsers import parse_pptx
        p = tmp_dir / "slides.pptx"
        p.write_bytes(b"fake pptx")
        mocker.patch.dict("sys.modules", {"pptx": None}, clear=False)
        result = parse_pptx(p)
        assert "python-pptx" in result.lower() or "pptx" in result.lower()


@pytest.mark.unit
class TestParseOdt:
    def test_valid_odt(self, mocker, tmp_dir):
        """ODT parsing from a minimal zip with content.xml."""
        import zipfile

        from inference_server.parsers import parse_odt
        p = tmp_dir / "doc.odt"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("content.xml", SAMPLE_ODT_ZIP_CONTENT)
        result = parse_odt(p)
        assert "Hello from ODT" in result

    def test_corrupt_odt(self, tmp_dir):
        from inference_server.parsers import parse_odt
        p = tmp_dir / "bad.odt"
        p.write_bytes(b"not a zip")
        result = parse_odt(p)
        assert "ODT:" in result and "failed" in result.lower()


@pytest.mark.unit
class TestParseRtf:
    def test_rtf_fallback_strip(self, tmp_dir):
        from inference_server.parsers import parse_rtf
        p = tmp_dir / "doc.rtf"
        p.write_text(SAMPLE_RTF)
        # May fail if striprtf is installed, but fallback should still work
        result = parse_rtf(p)
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.unit
class TestParseEpub:
    def test_epub_from_zip(self, tmp_dir):
        import zipfile

        from inference_server.parsers import parse_epub
        p = tmp_dir / "book.epub"
        html_content = "<html><body><p>Chapter 1 content</p></body></html>"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("chapter1.xhtml", html_content)
        result = parse_epub(p)
        assert "Chapter 1 content" in result

    def test_corrupt_epub(self, tmp_dir):
        from inference_server.parsers import parse_epub
        p = tmp_dir / "bad.epub"
        p.write_bytes(b"not a zip")
        result = parse_epub(p)
        assert "EPUB:" in result and "failed" in result.lower()


@pytest.mark.unit
class TestParseDocument:
    @pytest.mark.parametrize("ext,content,expected_substring", [
        (".txt", "Hello world", "Hello world"),
        (".md", "# Title\nBody", "Title"),
        (".json", '{"a": 1}', '"a"'),
        (".csv", "x,y\n1,2", "x | y"),
    ])
    def test_dispatches_by_extension(self, tmp_dir, ext, content, expected_substring):
        from inference_server.parsers import parse_document
        p = tmp_dir / f"test{ext}"
        p.write_text(content)
        result = parse_document(str(p))
        assert expected_substring in result

    def test_missing_file(self):
        from inference_server.parsers import parse_document
        result = parse_document("/nonexistent/file.txt")
        assert "not found" in result.lower()

    def test_unknown_extension_falls_back_to_text(self, tmp_dir):
        from inference_server.parsers import parse_document
        p = tmp_dir / "data.xyz123"
        p.write_text("fallback text")
        result = parse_document(str(p))
        assert "fallback text" in result

    def test_unknown_binary_fails_gracefully(self, tmp_dir):
        from inference_server.parsers import parse_document
        p = tmp_dir / "data.xyz123"
        p.write_bytes(b"\x00\x01\x02\x03")
        result = parse_document(str(p))
        # Should not raise, just return something
        assert isinstance(result, str)


@pytest.mark.unit
class TestParseBytes:
    def test_text_bytes(self):
        from inference_server.parsers import parse_bytes
        result = parse_bytes("hello.txt", b"Hello from bytes")
        assert "Hello from bytes" in result

    def test_json_bytes(self):
        from inference_server.parsers import parse_bytes
        data = json.dumps({"key": "val"}).encode()
        result = parse_bytes("data.json", data)
        parsed = json.loads(result)
        assert parsed["key"] == "val"

    def test_unknown_format(self):
        from inference_server.parsers import parse_bytes
        result = parse_bytes("file.xyz", b"some data")
        assert isinstance(result, str)


@pytest.mark.unit
class TestSupportedExtensions:
    def test_returns_sorted_list(self):
        from inference_server.parsers import supported_extensions
        exts = supported_extensions()
        assert isinstance(exts, list)
        assert len(exts) > 10
        assert exts == sorted(exts)
        assert ".txt" in exts
        assert ".pdf" in exts


@pytest.mark.unit
class TestIngestBytes:
    def test_ingest_valid_text(self, mocker, mock_rag_store):
        from inference_server.parsers import ingest_bytes
        store = mocker.MagicMock()
        store.add_document.return_value = 2
        result = ingest_bytes("doc.txt", b"Hello world this is test content", store)
        assert result["filename"] == "doc.txt"
        assert result["chunks"] == 2
        store.add_document.assert_called_once()

    def test_ingest_empty_content_returns_error(self, mocker):
        from inference_server.parsers import ingest_bytes
        store = mocker.MagicMock()
        result = ingest_bytes("doc.txt", b"", store)
        assert result["chunks"] == 0
        assert "error" in result
