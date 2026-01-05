"""Test agent - run directly to see agent execution trajectory."""

import argparse
import logging
import shutil
from pathlib import Path
from urllib.request import urlretrieve

from pangu_agent.agent import Agent
from pangu_agent.library.manager import LibraryManager
from pangu_agent.llm_client.client import LLMClient
from pangu_agent.prompts import (
    LITERATURE_ORGANIZER_SYSTEM_PROMPT,
    build_file_organization_prompt,
)
from pangu_agent.tools.explore_library import ExploreLibraryTool
from pangu_agent.tools.move_file import MoveFileTool
from pangu_agent.tools.view_file import ViewFileTool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

TEST_LIBRARY_ROOT = Path(__file__).parent.parent / "runs" / "test_library"


def setup_test_library():
    """Reset and create a fresh test library with sample files."""
    if TEST_LIBRARY_ROOT.exists():
        shutil.rmtree(TEST_LIBRARY_ROOT)
        print(f"✓ Removed existing test library")

    TEST_LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    (TEST_LIBRARY_ROOT / ".inbox").mkdir(exist_ok=True)
    # (TEST_LIBRARY_ROOT / "papers").mkdir(exist_ok=True)
    # (TEST_LIBRARY_ROOT / "papers" / "AI").mkdir(exist_ok=True)

    pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
    test_pdf = TEST_LIBRARY_ROOT / ".inbox" / "test_paper.pdf"
    print(f"  Downloading PDF from {pdf_url}...")
    try:
        urlretrieve(pdf_url, test_pdf)
        print(f"  ✓ Downloaded test_paper.pdf")
    except Exception as e:
        print(f"  ✗ Failed to download PDF: {e}")
        # Fallback to dummy content
        test_pdf.write_text("attention is all you need")

    test_meta = TEST_LIBRARY_ROOT / ".inbox" / ".test_paper.pdf.meta.json"
    test_meta.write_text('{"title": "Attention Is All You Need", "year": 2017, "authors": ["Vaswani et al."]}')

    print(f"✓ Test library created at: {TEST_LIBRARY_ROOT}")


def test_agent_explore_library():
    """Test agent with real LLM - explore library."""
    print("\n" + "=" * 60)
    print("TEST: Agent Explore Library")
    print("=" * 60)

    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    llm_client = LLMClient(log=False)
    tools = [
        ExploreLibraryTool(manager),
        ViewFileTool(manager),
        MoveFileTool(manager),
    ]

    agent = Agent(llm_client=llm_client, tools=tools, max_iterations=3)

    agent.add_system_prompt("You are a helpful assistant that can explore file systems.")
    agent.add_user_message("Please explore the library root directory and tell me what you find.")

    print("\n[Running agent...]")
    result = agent.run()

    print(f"\n[Result]")
    print(f"  Success: {result['success']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Stop reason: {result['stop_reason']}")
    if result.get('final_message'):
        print(f"  Final message: {result['final_message'][:200]}...")


def test_agent_organize_file():
    """Test agent with real LLM - organize a file (like add_literature)."""
    print("\n" + "=" * 60)
    print("TEST: Agent Organize File")
    print("=" * 60)

    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    llm_client = LLMClient(log=False)
    tools = [
        ExploreLibraryTool(manager),
        ViewFileTool(manager),
        MoveFileTool(manager),
    ]

    agent = Agent(llm_client=llm_client, tools=tools, max_iterations=5)

    # Read the test file
    inbox_path = Path(".inbox/test_paper.pdf")
    file_content = manager.read_file(str(inbox_path))

    agent.add_system_prompt(LITERATURE_ORGANIZER_SYSTEM_PROMPT)
    agent.add_user_message(build_file_organization_prompt(inbox_path, file_content))

    def stop_when_file_moved(context):
        return context["tool_name"] == "move_file" and context["result"].success

    print("\n[Running agent with stop condition...]")
    result = agent.run(stop_condition=stop_when_file_moved)

    print(f"\n[Result]")
    print(f"  Success: {result['success']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Stop reason: {result['stop_reason']}")
    if result.get('result'):
        print(f"  Moved to: {result['result']['tool_args'].get('dest_path')}")


def test_agent_memory():
    """Test agent memory management."""
    print("\n" + "=" * 60)
    print("TEST: Agent Memory")
    print("=" * 60)

    manager = LibraryManager(str(TEST_LIBRARY_ROOT))
    llm_client = LLMClient(log=False)
    tools = [ExploreLibraryTool(manager)]

    agent = Agent(llm_client=llm_client, tools=tools, max_iterations=5)

    print("\n[1] Initial state:")
    print(f"  Memory length: {len(agent.memory.history)}")

    print("\n[2] Add prompts:")
    agent.add_system_prompt("You are a library organizer.")
    agent.add_user_message("Please explore the library.")
    print(f"  Memory length: {len(agent.memory.history)}")

    print("\n[3] Memory contents:")
    for i, msg in enumerate(agent.memory.history):
        role = msg.get("role")
        content = msg.get("content", "")[:50]
        print(f"  [{i}] {role}: {content}...")


def main():
    parser = argparse.ArgumentParser(description="Test agent with configurable selection")
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["explore", "organize", "memory"],
        default=["explore", "organize", "memory"],
        help="Select which tests to run (default: all)",
    )
    args = parser.parse_args()

    print("Setting up test library...")
    setup_test_library()

    # Run selected tests
    if "memory" in args.tests:
        test_agent_memory()

    if "explore" in args.tests:
        test_agent_explore_library()

    if "organize" in args.tests:
        test_agent_organize_file()

    print("\n" + "=" * 60)
    print("All agent tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
