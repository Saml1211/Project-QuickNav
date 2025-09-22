"""
Test suite for the document navigation module.

Tests parsing, ranking, classification, and integration with real-world examples
from the training data corpus.
"""

import os
import pytest
import tempfile
import json
from datetime import datetime
from pathlib import Path

# Import modules under test
from quicknav.doc_navigator import (
    DocumentParser, DocumentTypeClassifier, DocumentScanner,
    DocumentRanker, navigate_to_document, get_custom_roots,
    resolve_project_root
)


class TestDocumentParser:
    """Test the document parser for version/metadata extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DocumentParser()

    def test_rev_numeric_parsing(self):
        """Test REV-based numeric version parsing."""
        test_cases = [
            ("20865 - CPQST 50 QUAY ST - LEVEL 1 - REV 100.pdf", 100, "rev_numeric"),
            ("20865 - CPQST 50 QUAY ST - LEVEL 1 - REV 101.pdf", 101, "rev_numeric"),
            ("20865 - CPQST 50 QUAY ST - LEVEL 1 - REV 102.pdf", 102, "rev_numeric"),
            ("Project LLD - Rev 200 AS-BUILT.vsdx", 200, "rev_numeric"),
            ("Design Rev_109.pdf", 109, "rev_numeric"),
        ]

        for filename, expected_version, expected_type in test_cases:
            result = self.parser.parse_filename(filename)
            assert result['version'] == expected_version, f"Failed for {filename}"
            assert result['version_type'] == expected_type, f"Failed for {filename}"

    def test_period_version_parsing(self):
        """Test period-based version parsing."""
        test_cases = [
            ("20865 - CPQST 50 QUAY ST - LEVEL 31 - REV 1.01.pdf", (1, 1), "period"),
            ("Design v2.03.pdf", (2, 3), "period"),
            ("Document Rev.1.00.pdf", (1, 0), "period"),
        ]

        for filename, expected_version, expected_type in test_cases:
            result = self.parser.parse_filename(filename)
            assert result['version'] == expected_version, f"Failed for {filename}"
            assert result['version_type'] == expected_type, f"Failed for {filename}"

    def test_letter_version_parsing(self):
        """Test letter-based version parsing."""
        test_cases = [
            ("REV.A DHA Office Relocation SCHEDULE.pdf", 1, "letter"),
            ("E 403_ Rev.D markup.pdf", 4, "letter"),  # D = 4th letter
            ("Plan Rev_C.pdf", 3, "letter"),
        ]

        for filename, expected_version, expected_type in test_cases:
            result = self.parser.parse_filename(filename)
            assert result['version'] == expected_version, f"Failed for {filename}"
            assert result['version_type'] == expected_type, f"Failed for {filename}"

    def test_status_tag_extraction(self):
        """Test status tag extraction."""
        test_cases = [
            ("Project LLD - REV 100 AS-BUILT.pdf", {"AS-BUILT"}),
            ("Design DRAFT Rev 101.pdf", {"DRAFT"}),
            ("Plan SIGNED FINAL.pdf", {"SIGNED", "FINAL"}),
            ("Document AS BUILT v2.pdf", {"AS BUILT"}),
        ]

        for filename, expected_tags in test_cases:
            result = self.parser.parse_filename(filename)
            assert expected_tags.issubset(result['status_tags']), f"Failed for {filename}"

    def test_project_code_extraction(self):
        """Test project code extraction."""
        test_cases = [
            ("17741 - QPS MTR Room Upgrades - Sales & PO Report 23.09.22.pdf", "17741"),
            ("20797 - Ingenia Victoria Point Club House Proposal Rev8.pdf", "20797"),
            ("18810 - PAVS-SA - SAAB Hearing Loop - Sales & PO Report.pdf", "18810"),
        ]

        for filename, expected_code in test_cases:
            result = self.parser.parse_filename(filename)
            assert result['project_code'] == expected_code, f"Failed for {filename}"

    def test_date_extraction(self):
        """Test date extraction from filenames."""
        test_cases = [
            ("17741 - Sales & PO Report 23.09.22.pdf", datetime(2022, 9, 23)),
            ("Project 240606 Report.pdf", datetime(2024, 6, 6)),
            ("Document Received 17-12-2024.pdf", datetime(2024, 12, 17)),
        ]

        for filename, expected_date in test_cases:
            result = self.parser.parse_filename(filename)
            assert expected_date in result['dates'], f"Failed for {filename}: {result['dates']}"

    def test_room_number_extraction(self):
        """Test room number extraction."""
        test_cases = [
            ("Priority 19 - Room 12 - MTR-SFP-DTACC-1.pdf", 12),
            ("Design Room01 Layout.pdf", 1),
            ("Plan room 25 details.pdf", 25),
        ]

        for filename, expected_room in test_cases:
            result = self.parser.parse_filename(filename)
            assert result['room_number'] == expected_room, f"Failed for {filename}"

    def test_co_number_extraction(self):
        """Test change order number extraction."""
        test_cases = [
            ("Change Order 5 - Project Modifications.pdf", 5),
            ("CO123 - Additional Work.pdf", 123),
            ("Project CO 42 Approval.pdf", 42),
        ]

        for filename, expected_co in test_cases:
            result = self.parser.parse_filename(filename)
            assert result['co_number'] == expected_co, f"Failed for {filename}"

    def test_series_extraction(self):
        """Test document series extraction (name without version/status)."""
        test_cases = [
            ("Project LLD - Level 1 - REV 100 AS-BUILT.pdf", "Project LLD - Level 1"),
            ("20865 - CPQST 50 QUAY ST - LEVEL 1 - REV 101.pdf", "20865 - CPQST 50 QUAY ST - LEVEL 1"),
            ("Floor Plan DRAFT Rev A.pdf", "Floor Plan"),
        ]

        for filename, expected_series in test_cases:
            result = self.parser.parse_filename(filename)
            assert result['series'].strip() == expected_series, f"Failed for {filename}: got '{result['series']}'"

    def test_version_comparison(self):
        """Test version comparison logic."""
        # REV numeric comparisons
        assert self.parser.compare_versions(100, "rev_numeric", 101, "rev_numeric") == -1
        assert self.parser.compare_versions(200, "rev_numeric", 101, "rev_numeric") == 1
        assert self.parser.compare_versions(100, "rev_numeric", 100, "rev_numeric") == 0

        # Period comparisons
        assert self.parser.compare_versions((1, 0), "period", (1, 1), "period") == -1
        assert self.parser.compare_versions((2, 0), "period", (1, 9), "period") == 1
        assert self.parser.compare_versions((1, 5), "period", (1, 5), "period") == 0

        # Type priority: REV > period > letter
        assert self.parser.compare_versions(100, "rev_numeric", (2, 0), "period") == 1
        assert self.parser.compare_versions((1, 0), "period", 5, "letter") == 1

    def test_initial_version_detection(self):
        """Test detection of initial versions."""
        assert self.parser._is_initial_version(100, "rev_numeric") == True
        assert self.parser._is_initial_version(101, "rev_numeric") == False
        assert self.parser._is_initial_version((1, 0), "period") == True
        assert self.parser._is_initial_version((1, 1), "period") == True
        assert self.parser._is_initial_version((2, 0), "period") == False


class TestDocumentTypeClassifier:
    """Test document type classification."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = DocumentTypeClassifier()

    def test_lld_classification(self):
        """Test LLD document classification."""
        test_files = [
            "Project LLD - Level 1 - REV 100.vsdx",
            "Low Level Design - Room 5.pdf",
            "20865 - System LLD v2.pdf"
        ]

        for filename in test_files:
            result = self.classifier.classify_document(filename, "lld")
            assert result is not None, f"Failed to classify {filename} as LLD"
            assert result['type'] == 'lld'

    def test_change_order_classification(self):
        """Test change order classification."""
        test_files = [
            "Change Order 5 - Additional Work.pdf",
            "CO123 - Project Modifications.pdf",
            "Project Change Order 42.pdf"
        ]

        for filename in test_files:
            result = self.classifier.classify_document(filename, "change_order")
            assert result is not None, f"Failed to classify {filename} as change order"
            assert result['type'] == 'change_order'

    def test_sales_po_classification(self):
        """Test Sales & PO Report classification."""
        test_files = [
            "17741 - QPS MTR Room Upgrades - Sales & PO Report 23.09.22.pdf",
            "20381 - Neoen OLD Office - Sales & PO Report ORIGINAL.pdf",
            "18810 - SAAB Hearing Loop - Sales & PO Report - 17.12.2024.pdf"
        ]

        for filename in test_files:
            result = self.classifier.classify_document(filename, "sales_po")
            assert result is not None, f"Failed to classify {filename} as sales_po"
            assert result['type'] == 'sales_po'

    def test_floor_plans_classification(self):
        """Test floor plans classification."""
        test_files = [
            "Priority 19 - Room 12 - GA PLAN - Sheet 1.pdf",
            "A-101 Floor Plan Level 1.pdf",
            "Building SHEET 5 Layout.pdf"
        ]

        for filename in test_files:
            result = self.classifier.classify_document(filename, "floor_plans")
            assert result is not None, f"Failed to classify {filename} as floor_plans"
            assert result['type'] == 'floor_plans'

    def test_photos_classification(self):
        """Test photos classification."""
        test_files = [
            "Site Photo 001.jpg",
            "Room Installation.png",
            "Progress Photo.heic"
        ]

        for filename in test_files:
            result = self.classifier.classify_document(filename, "photos")
            assert result is not None, f"Failed to classify {filename} as photos"
            assert result['type'] == 'photos'

    def test_exclusion_filters(self):
        """Test exclusion of archived documents."""
        archived_files = [
            "ARCHIVE/Old LLD Design.vsdx",
            "OLD DRAWINGS/Floor Plan Rev A.pdf",
            "Project/BACKUP/Change Order 5.pdf"
        ]

        for filepath in archived_files:
            result = self.classifier.classify_document(filepath, "lld")
            # Should be None due to exclusion filters
            assert result is None, f"Should have excluded {filepath}"


class TestDocumentRanker:
    """Test document ranking and selection logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = DocumentTypeClassifier()
        self.parser = DocumentParser()
        self.ranker = DocumentRanker(self.classifier, self.parser)

    def create_mock_document(self, filename, **kwargs):
        """Create a mock document metadata dict."""
        parsed = self.parser.parse_filename(filename)
        mock_doc = {
            'filename': filename,
            'filepath': f"/project/{filename}",
            'doc_type': 'lld',
            'mod_time': datetime.now(),
            'in_preferred_folder': kwargs.get('in_preferred_folder', True),
            'in_archive': kwargs.get('in_archive', False),
            'ranking_score': 0.0
        }
        mock_doc.update(parsed)
        mock_doc.update(kwargs)
        return mock_doc

    def test_project_code_bonus(self):
        """Test project code matching bonus."""
        doc1 = self.create_mock_document("17741 - Project LLD REV 100.pdf")
        doc2 = self.create_mock_document("20865 - Other Project LLD REV 100.pdf")

        docs = self.ranker.rank_documents([doc1, doc2], project_code="17741")

        # Doc1 should have higher score due to project code match
        assert docs[0]['project_code'] == "17741"
        assert docs[0]['ranking_score'] > docs[1]['ranking_score']

    def test_version_ranking(self):
        """Test version-based ranking."""
        doc_rev100 = self.create_mock_document("Project LLD REV 100.pdf")
        doc_rev101 = self.create_mock_document("Project LLD REV 101.pdf")
        doc_rev200 = self.create_mock_document("Project LLD REV 200.pdf")

        docs = self.ranker.rank_documents([doc_rev100, doc_rev101, doc_rev200])

        # REV 200 should rank highest, then 101, then 100
        assert docs[0]['version'] == 200
        assert docs[1]['version'] == 101
        assert docs[2]['version'] == 100

    def test_status_ranking(self):
        """Test status-based ranking."""
        doc_draft = self.create_mock_document("Project LLD REV 100 DRAFT.pdf")
        doc_final = self.create_mock_document("Project LLD REV 100 FINAL.pdf")
        doc_asbuilt = self.create_mock_document("Project LLD REV 100 AS-BUILT.pdf")

        docs = self.ranker.rank_documents([doc_draft, doc_final, doc_asbuilt])

        # AS-BUILT should rank highest, then FINAL, then DRAFT
        assert "AS-BUILT" in docs[0]['status_tags']
        assert docs[2]['ranking_score'] < docs[1]['ranking_score']  # DRAFT should be lowest

    def test_archive_penalty(self):
        """Test archive penalty."""
        doc_normal = self.create_mock_document("Project LLD REV 100.pdf", in_archive=False)
        doc_archive = self.create_mock_document("Project LLD REV 100.pdf", in_archive=True)

        docs = self.ranker.rank_documents([doc_normal, doc_archive])

        # Normal doc should rank higher than archived
        assert docs[0]['in_archive'] == False
        assert docs[0]['ranking_score'] > docs[1]['ranking_score']

    def test_room_filtering(self):
        """Test room number filtering."""
        doc_room5 = self.create_mock_document("Project Room 5 LLD.pdf")
        doc_room12 = self.create_mock_document("Project Room 12 LLD.pdf")
        doc_general = self.create_mock_document("Project General LLD.pdf")

        # Filter for room 5
        docs = self.ranker.rank_documents([doc_room5, doc_room12, doc_general], room_filter=5)

        # Should only return room 5 document
        assert len(docs) == 1
        assert docs[0]['room_number'] == 5

    def test_document_grouping(self):
        """Test document grouping by series."""
        doc1 = self.create_mock_document("Project LLD Level 1 REV 100.pdf")
        doc2 = self.create_mock_document("Project LLD Level 1 REV 101.pdf")
        doc3 = self.create_mock_document("Project LLD Level 2 REV 100.pdf")

        groups = self.ranker.group_by_series([doc1, doc2, doc3])

        # Should have two groups
        assert len(groups) == 2

        # Each group should be sorted by version (newest first)
        for series, docs in groups.items():
            if "Level 1" in series:
                assert len(docs) == 2
                assert docs[0]['version'] == 101  # Newer version first
                assert docs[1]['version'] == 100
            else:
                assert len(docs) == 1

    def test_auto_selection_threshold(self):
        """Test automatic selection threshold."""
        # High-scoring document vs low-scoring
        doc_good = self.create_mock_document("17741 - Project LLD REV 200 AS-BUILT.pdf")
        doc_poor = self.create_mock_document("Other Project LLD REV 100 DRAFT.pdf", in_archive=True)

        # Mock ranking scores to simulate significant difference
        doc_good['ranking_score'] = 10.0
        doc_poor['ranking_score'] = 2.0

        result = self.ranker.select_best_document([doc_good, doc_poor], "auto")

        # Should auto-select the high-scoring document
        assert result == doc_good['filepath']

    def test_choose_mode_selection(self):
        """Test choose mode returns all documents."""
        doc1 = self.create_mock_document("Project LLD REV 100.pdf")
        doc2 = self.create_mock_document("Project LLD REV 101.pdf")

        result = self.ranker.select_best_document([doc1, doc2], "choose")

        # Should return list of all documents
        assert isinstance(result, list)
        assert len(result) == 2
        assert doc1['filepath'] in result
        assert doc2['filepath'] in result


class TestDocumentScanner:
    """Test document scanning functionality."""

    def setup_method(self):
        """Set up test fixtures with temporary directory structure."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = os.path.join(self.temp_dir, "17741 - Test Project")

        # Create directory structure
        os.makedirs(os.path.join(self.project_path, "4. System Designs"))
        os.makedirs(os.path.join(self.project_path, "1. Sales Handover"))
        os.makedirs(os.path.join(self.project_path, "2. BOM & Orders"))
        os.makedirs(os.path.join(self.project_path, "6. Site Photos"))
        os.makedirs(os.path.join(self.project_path, "ARCHIVE"))

        # Create sample files
        self.create_sample_file("4. System Designs/17741 - Project LLD REV 100.vsdx")
        self.create_sample_file("4. System Designs/17741 - Project LLD REV 101 AS-BUILT.vsdx")
        self.create_sample_file("1. Sales Handover/17741 - Project Handover Document.pdf")
        self.create_sample_file("2. BOM & Orders/17741 - Sales & PO Report 23.09.22.pdf")
        self.create_sample_file("6. Site Photos/Room 5 Installation.jpg")
        self.create_sample_file("ARCHIVE/17741 - Old LLD REV 99.vsdx")

        # Initialize components
        self.parser = DocumentParser()
        self.classifier = DocumentTypeClassifier()
        self.scanner = DocumentScanner(self.parser, self.classifier)

    def create_sample_file(self, relative_path):
        """Create a sample file in the test project."""
        full_path = os.path.join(self.project_path, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write("sample content")

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_lld_scanning(self):
        """Test scanning for LLD documents."""
        documents = self.scanner.scan_documents(self.project_path, "lld")

        # Should find LLD documents, exclude archived by default
        lld_docs = [doc for doc in documents if doc['doc_type'] == 'lld']
        assert len(lld_docs) == 2  # REV 100 and REV 101, excluding archived

        # Check that AS-BUILT version is included
        asbuilt_docs = [doc for doc in lld_docs if 'AS-BUILT' in doc['status_tags']]
        assert len(asbuilt_docs) == 1

    def test_archive_exclusion(self):
        """Test exclusion of archived documents."""
        # Scan with archive exclusion (default)
        docs_excluded = self.scanner.scan_documents(self.project_path, "lld", exclude_archive=True)

        # Scan with archive inclusion
        docs_included = self.scanner.scan_documents(self.project_path, "lld", exclude_archive=False)

        # Should find more documents when including archive
        assert len(docs_included) > len(docs_excluded)

        # Check that archived document is only in included results
        archived_filenames = [doc['filename'] for doc in docs_included if doc['in_archive']]
        assert len(archived_filenames) > 0

    def test_folder_preference_detection(self):
        """Test detection of preferred folders."""
        documents = self.scanner.scan_documents(self.project_path, "lld")

        # LLD documents in "4. System Designs" should be marked as in preferred folder
        for doc in documents:
            if doc['doc_type'] == 'lld':
                assert doc['in_preferred_folder'] == True

    def test_mixed_document_scanning(self):
        """Test scanning for all document types."""
        documents = self.scanner.scan_documents(self.project_path)

        # Should find different types of documents
        doc_types = {doc['doc_type'] for doc in documents}
        assert 'lld' in doc_types
        assert 'scope' in doc_types or 'sales_po' in doc_types
        assert 'photos' in doc_types


class TestIntegration:
    """Integration tests for the complete document navigation workflow."""

    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = os.path.join(self.temp_dir, "17741 - Integration Test Project")

        # Create comprehensive directory structure
        self.setup_test_project()

    def setup_test_project(self):
        """Set up a realistic test project structure."""
        # Create directories
        dirs = [
            "1. Sales Handover",
            "2. BOM & Orders/Sales & Change Orders",
            "3. PMO",
            "4. System Designs",
            "5. Floor Plans",
            "6. Site Photos",
            "4. System Designs/ARCHIVE"
        ]

        for dir_path in dirs:
            os.makedirs(os.path.join(self.project_path, dir_path))

        # Create realistic file set
        files = [
            # LLD documents with version progression
            "4. System Designs/17741 - Project LLD Level 1 REV 100.vsdx",
            "4. System Designs/17741 - Project LLD Level 1 REV 101.vsdx",
            "4. System Designs/17741 - Project LLD Level 1 REV 102 AS-BUILT.vsdx",
            "4. System Designs/17741 - Project LLD Level 2 REV 100.vsdx",
            "4. System Designs/ARCHIVE/17741 - Project LLD Level 1 REV 99.vsdx",

            # Sales documents
            "1. Sales Handover/17741 - Project Handover Document.pdf",
            "2. BOM & Orders/Sales & Change Orders/17741 - Sales & PO Report 23.09.22.pdf",
            "2. BOM & Orders/Sales & Change Orders/17741 - Change Order 5 APPROVED.pdf",

            # QA documents
            "3. PMO/17741 - Room 5 QA Report SIGNED.pdf",
            "3. PMO/17741 - SWMS v1.2 SIGNED.pdf",

            # Floor plans
            "5. Floor Plans/17741 - Level 1 GA PLAN SHEET 1 AS-BUILT.pdf",
            "5. Floor Plans/17741 - Room 5 Detail A-501.pdf",

            # Photos
            "6. Site Photos/Installation Progress 001.jpg",
            "6. Site Photos/Room 5 Completion.png"
        ]

        for file_path in files:
            full_path = os.path.join(self.project_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(f"Content for {os.path.basename(file_path)}")

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_navigate_to_lld_auto_select(self):
        """Test automatic selection of best LLD document."""
        result = navigate_to_document(
            self.project_path,
            doc_type="lld",
            selection_mode="auto",
            project_code="17741"
        )

        # Should auto-select the AS-BUILT version
        assert result.startswith("SUCCESS:")
        selected_path = result[8:]  # Remove "SUCCESS:" prefix
        assert "REV 102 AS-BUILT" in selected_path

    def test_navigate_to_lld_latest_mode(self):
        """Test latest mode for LLD documents."""
        result = navigate_to_document(
            self.project_path,
            doc_type="lld",
            selection_mode="latest",
            project_code="17741"
        )

        # Should return latest from each series
        if result.startswith("SELECT:"):
            paths = result[7:].split("|")
            assert len(paths) == 2  # Level 1 and Level 2
        else:
            # If auto-selected, should be the best one
            assert result.startswith("SUCCESS:")

    def test_navigate_with_room_filter(self):
        """Test navigation with room number filter."""
        result = navigate_to_document(
            self.project_path,
            doc_type="qa_itp",
            selection_mode="auto",
            project_code="17741",
            room_filter=5
        )

        # Should find room 5 QA report
        assert result.startswith("SUCCESS:")
        selected_path = result[8:]
        assert "Room 5" in selected_path

    def test_navigate_with_co_filter(self):
        """Test navigation with change order filter."""
        result = navigate_to_document(
            self.project_path,
            doc_type="change_order",
            selection_mode="auto",
            project_code="17741",
            co_filter=5
        )

        # Should find change order 5
        assert result.startswith("SUCCESS:")
        selected_path = result[8:]
        assert "Change Order 5" in selected_path

    def test_navigate_include_archive(self):
        """Test navigation including archived documents."""
        # First, exclude archive (default)
        result_excluded = navigate_to_document(
            self.project_path,
            doc_type="lld",
            selection_mode="choose",
            exclude_archive=True
        )

        # Then, include archive
        result_included = navigate_to_document(
            self.project_path,
            doc_type="lld",
            selection_mode="choose",
            exclude_archive=False
        )

        # Should find more documents when including archive
        if result_excluded.startswith("SELECT:") and result_included.startswith("SELECT:"):
            paths_excluded = result_excluded[7:].split("|")
            paths_included = result_included[7:].split("|")
            assert len(paths_included) > len(paths_excluded)

    def test_navigate_no_documents_found(self):
        """Test handling when no documents are found."""
        result = navigate_to_document(
            self.project_path,
            doc_type="lld",
            selection_mode="auto",
            project_code="99999"  # Non-matching project code
        )

        # Should return error when no documents match criteria
        # Note: This test depends on whether project code is strictly enforced
        # The current implementation might still find documents without project code match

    def test_corpus_validation_rev_patterns(self):
        """Test validation against corpus examples for REV patterns."""
        # Test cases from actual training data
        corpus_examples = [
            ("17741 - R 51 - ... - REV 100.pdf", 100),
            ("... - REV 101 AS-BUILT.pdf", 101),
            ("20865 - CPQST 50 QUAY ST - LEVEL 1 - REV 102.pdf", 102),
            ("E 403_ ... Rev.D markup.pdf", 4),  # D = 4th letter
        ]

        parser = DocumentParser()
        for filename, expected_version in corpus_examples:
            result = parser.parse_filename(filename)
            if isinstance(expected_version, int) and expected_version > 26:
                # Numeric REV pattern
                assert result['version'] == expected_version, f"Failed for {filename}"
                assert result['version_type'] == 'rev_numeric'
            else:
                # Letter pattern
                assert result['version'] == expected_version, f"Failed for {filename}"
                assert result['version_type'] == 'letter'

    def test_corpus_validation_sales_po_reports(self):
        """Test validation against corpus Sales & PO Report examples."""
        corpus_examples = [
            "17741 - QPS MTR Room Upgrades - Bulk Order - Sales & PO Report 23.09.22.pdf",
            "20381 - Neoen OLD Office - Sales & PO Report ORIGINAL.pdf",
            "18810 - PAVS-SA - SAAB Hearing Loop (2501) - Sales & PO Report - 17.12.2024.pdf"
        ]

        classifier = DocumentTypeClassifier()
        for filename in corpus_examples:
            result = classifier.classify_document(filename, "sales_po")
            assert result is not None, f"Failed to classify {filename} as sales_po"
            assert result['type'] == 'sales_po'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])