"""Test tools - run directly to see outputs."""

import argparse
import logging
import shutil
from pathlib import Path
from urllib.request import urlretrieve

from pangu_agent.library.manager import LibraryManager
from pangu_agent.tools import (
    ExploreLibraryTool,
    FinishTool,
    MoveFileTool,
    SearchLibraryTool,
    ToolExecutor,
    ViewFileTool,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

TEST_LIBRARY_ROOT = Path(__file__).parent.parent / "runs" / "test_library"


def print_result(result):
    """Print tool execution result in a unified format."""
    if result.success:
        print(f"✓ Success")
        print(f"Output: {result.output}")
    else:
        print(f"✗ Failed")
        print(f"Error: {result.error}")


def setup_test_library():
    """Reset and create a fresh test library with sample structure."""
    # Remove existing library
    if TEST_LIBRARY_ROOT.exists():
        shutil.rmtree(TEST_LIBRARY_ROOT)
        print(f"✓ Removed existing test library")

    # Create fresh structure
    TEST_LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    (TEST_LIBRARY_ROOT / ".inbox").mkdir(exist_ok=True)
    (TEST_LIBRARY_ROOT / "papers").mkdir(exist_ok=True)
    (TEST_LIBRARY_ROOT / "papers" / "AI").mkdir(exist_ok=True)
    (TEST_LIBRARY_ROOT / "books").mkdir(exist_ok=True)

    # Download real PDF - Attention Is All You Need paper
    pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
    test_pdf = TEST_LIBRARY_ROOT / "papers" / "AI" / "transformer.pdf"

    print(f"  Downloading PDF from {pdf_url}...")
    try:
        urlretrieve(pdf_url, test_pdf)
        print(f"  ✓ Downloaded transformer.pdf")
    except Exception as e:
        print(f"  ✗ Failed to download PDF: {e}")
        # Fallback to dummy content
        test_pdf.write_text("attention is all you need")

    # Create metadata file for the PDF
    test_meta = TEST_LIBRARY_ROOT / "papers" / "AI" / ".transformer.pdf.meta.json"
    test_meta.write_text('{"title": "Attention Is All You Need", "year": 2017, "authors": ["Vaswani et al."]}')

    # Download real image - use a reliable public source
    img_url = "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-dark.png"
    test_img = TEST_LIBRARY_ROOT / "papers" / "AI" / "sample_image.png"

    print(f"  Downloading image from {img_url}...")
    try:
        urlretrieve(img_url, test_img)
        print(f"  ✓ Downloaded sample_image.png")
    except Exception as e:
        print(f"  ✗ Failed to download image: {e}")

    (TEST_LIBRARY_ROOT / ".inbox" / "test.pdf").write_text("dummy pdf content")

    print(f"✓ Test library created at: {TEST_LIBRARY_ROOT}")


def test_explore_library_tool():
    """Test ExploreLibraryTool - browse directory structure."""
    print("\n" + "=" * 60)
    print("TEST: ExploreLibraryTool")
    print("=" * 60)

    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    tools = [ExploreLibraryTool(manager)]
    executor = ToolExecutor(tools)

    # Test 1: Explore root
    print("\n[1] Explore root directory:")
    result = executor.execute("explore_library", {"path": "."})
    print_result(result)

    # Test 2: Explore specific subdirectory
    print("\n[2] Explore papers directory:")
    result = executor.execute("explore_library", {"path": "papers"})
    print_result(result)

    # Test 3: Non-existent path
    print("\n[3] Explore non-existent directory:")
    result = executor.execute("explore_library", {"path": "nonexistent"})
    print_result(result)


def test_move_file_tool():
    """Test MoveFileTool - move files around."""
    print("\n" + "=" * 60)
    print("TEST: MoveFileTool")
    print("=" * 60)

    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    tools = [MoveFileTool(manager)]
    executor = ToolExecutor(tools)

    # Setup: ensure test file exists
    test_file = TEST_LIBRARY_ROOT / ".inbox" / "test_move.pdf"
    test_file.write_text("test content")

    # Test 1: Move to existing directory
    print("\n[1] Move file to existing directory:")
    result = executor.execute("move_file", {
        "source_path": ".inbox/test_move.pdf",
        "dest_path": "papers/test_move.pdf"
    })
    print_result(result)
    if (TEST_LIBRARY_ROOT / "papers" / "test_move.pdf").exists():
        print("  (verified: file exists at destination)")

    # Test 2: Move to new directory (should create it)
    test_file2 = TEST_LIBRARY_ROOT / ".inbox" / "test_move2.pdf"
    test_file2.write_text("test content 2")

    print("\n[2] Move file and create new directory:")
    result = executor.execute("move_file", {
        "source_path": ".inbox/test_move2.pdf",
        "dest_path": "papers/ML/test_move2.pdf"
    })
    print_result(result)

    # Test 3: Move non-existent file
    print("\n[3] Try to move non-existent file:")
    result = executor.execute("move_file", {
        "source_path": ".inbox/nonexistent.pdf",
        "dest_path": "papers/fail.pdf"
    })
    print_result(result)


def test_view_file_tool():
    """Test ViewFileTool - read file contents."""
    print("\n" + "=" * 60)
    print("TEST: ViewFileTool")
    print("=" * 60)

    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    tools = [ViewFileTool(manager)]
    executor = ToolExecutor(tools)

    # Test 1: View existing PDF file
    print("\n[1] View existing PDF file:")
    result = executor.execute("view_file", {"file_path": "papers/AI/transformer.pdf"})
    result.output["text"] = result.output["text"][:100] + "..."  # Truncate for display
    print_result(result)

    # Test 2: View existing image file
    print("\n[2] View existing image file:")
    result = executor.execute("view_file", {"file_path": "papers/AI/sample_image.png", "info_type": "content"})
    result.output["image_url"]["url"] = result.output["image_url"]["url"][:100] + "..."  # Truncate for display
    print_result(result)

    # Test 3: View non-existent file
    print("\n[3] View non-existent file:")
    result = executor.execute("view_file", {"file_path": "nonexistent.pdf"})
    print_result(result)


def test_search_library_tool():
    """Test SearchLibraryTool - semantic search using embeddings."""
    print("\n" + "=" * 60)
    print("TEST: SearchLibraryTool")
    print("=" * 60)

    # Initialize manager (this will also initialize embedding system)
    print("\nInitializing embedding system...")
    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    tools = [SearchLibraryTool(manager)]
    executor = ToolExecutor(tools)

    # First, stage some files to build the index
    print("\n[Setup] Staging files to build vector index...")

    # Stage the transformer PDF if it exists
    transformer_pdf = TEST_LIBRARY_ROOT / "papers" / "AI" / "transformer.pdf"
    if transformer_pdf.exists():
        try:
            staged = manager.stage_copy(str(transformer_pdf))
            print(f"  ✓ Staged: {staged.name}")
        except Exception as e:
            print(f"  ✗ Failed to stage transformer.pdf: {e}")

    # Stage the image if it exists
    sample_img = TEST_LIBRARY_ROOT / "papers" / "AI" / "sample_image.png"
    if sample_img.exists():
        try:
            staged = manager.stage_copy(str(sample_img))
            print(f"  ✓ Staged: {staged.name}")
        except Exception as e:
            print(f"  ✗ Failed to stage sample_image.png: {e}")

    # Check vector store count
    count = manager._vector_store.count()
    print(f"\n  Vector store now contains {count} embedding(s)")

    if count == 0:
        print("\n  ⚠ No embeddings in store, search tests will return empty results")

    # Test 1: Search with default (both types, grouped)
    print("\n[1] Search for 'transformer attention' (both types, top_k=2):")
    result = executor.execute("search_library", {
        "query": "transformer attention mechanism",
        "top_k": 2
    })
    print_result(result)

    # Test 2: Search only PDFs
    print("\n[2] Search for 'neural network' (only PDFs):")
    result = executor.execute("search_library", {
        "query": "neural network",
        "top_k": 2,
        "file_types": ["pdf"]
    })
    print_result(result)

    # Test 3: Search only images
    print("\n[3] Search for 'pytorch logo' (only images):")
    result = executor.execute("search_library", {
        "query": "pytorch logo",
        "top_k": 2,
        "file_types": ["image"]
    })
    print_result(result)

    # Test 4: Empty query should fail
    print("\n[4] Search with empty query (should fail):")
    result = executor.execute("search_library", {"query": ""})
    print_result(result)


def test_finish_tool():
    """Test FinishTool - terminate with result."""
    print("\n" + "=" * 60)
    print("TEST: FinishTool")
    print("=" * 60)

    tools = [FinishTool()]
    executor = ToolExecutor(tools)

    # Test 1: Finish with success status
    print("\n[1] Finish with success status:")
    result = executor.execute("finish", {
        "result": "Task completed successfully! Found 5 relevant papers.",
        "status": "success"
    })
    print_result(result)
    if result.success and isinstance(result.output, dict):
        print(f"  Finished: {result.output.get('finished')}")
        print(f"  Status: {result.output.get('status')}")
        print(f"  Result: {result.output.get('result')}")

    # Test 2: Finish with default status (success)
    print("\n[2] Finish with default status:")
    result = executor.execute("finish", {
        "result": "Analysis complete. The transformer architecture shows..."
    })
    print_result(result)
    if result.success and isinstance(result.output, dict):
        print(f"  Status (default): {result.output.get('status')}")

    # Test 3: Finish with partial status
    print("\n[3] Finish with partial completion:")
    result = executor.execute("finish", {
        "result": "Found 3 out of 5 requested papers. The remaining 2 were not available.",
        "status": "partial"
    })
    print_result(result)

    # Test 4: Finish with failed status
    print("\n[4] Finish with failed status:")
    result = executor.execute("finish", {
        "result": "Could not complete the task due to missing dependencies.",
        "status": "failed"
    })
    print_result(result)

    # Test 5: Empty result (should fail)
    print("\n[5] Finish with empty result (should fail):")
    result = executor.execute("finish", {"result": ""})
    print_result(result)

    # Test 6: Invalid status (should fail)
    print("\n[6] Finish with invalid status (should fail):")
    result = executor.execute("finish", {
        "result": "Some result",
        "status": "invalid_status"
    })
    print_result(result)


def main():
    parser = argparse.ArgumentParser(description="Test tools with configurable selection")
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=["explore", "move", "view", "search", "finish"],
        default=["explore", "move", "view", "search", "finish"],
        help="Select which tools to test (default: all)",
    )
    args = parser.parse_args()

    print("Setting up test library...")
    setup_test_library()

    # Run selected tests
    if "explore" in args.tools:
        test_explore_library_tool()

    if "move" in args.tools:
        test_move_file_tool()

    if "view" in args.tools:
        test_view_file_tool()

    if "search" in args.tools:
        test_search_library_tool()

    if "finish" in args.tools:
        test_finish_tool()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
