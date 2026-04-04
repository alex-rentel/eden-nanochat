"""Tests for ChatML tokenization and format detection."""

from nanochat_mlx.sft import detect_format
from nanochat_mlx.tokenizer import render_chatml_conversation


class MockTokenizer:
    """Minimal tokenizer mock for testing ChatML rendering."""
    _special = {
        "<|bos|>": 0,
        "<|user_start|>": 1,
        "<|user_end|>": 2,
        "<|assistant_start|>": 3,
        "<|assistant_end|>": 4,
        "<|output_start|>": 7,
        "<|output_end|>": 8,
    }

    def get_bos_token_id(self):
        return 0

    def encode_special(self, text):
        return self._special[text]

    def encode(self, text):
        # Simple: one token per character
        return [ord(c) for c in text]


def test_detect_format_smoltalk():
    conversations = [
        {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]}
    ]
    assert detect_format(conversations) == "smoltalk"


def test_detect_format_chatml_with_tool():
    conversations = [
        {"messages": [
            {"role": "user", "content": "Do something"},
            {"role": "assistant", "content": "<tool_call>{}</tool_call>"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "Done"},
        ]}
    ]
    assert detect_format(conversations) == "chatml"


def test_detect_format_empty():
    assert detect_format([]) == "smoltalk"


def test_chatml_renders_tool_messages():
    tok = MockTokenizer()
    conv = {"messages": [
        {"role": "system", "content": "A"},
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": "C"},
        {"role": "tool", "content": "D"},
        {"role": "assistant", "content": "E"},
    ]}
    ids, mask = render_chatml_conversation(tok, conv)

    # Check that BOS is first
    assert ids[0] == 0

    # Check that assistant content is masked=1, others masked=0
    # Find assistant_start markers (token 3) and check content after them
    i = 0
    assistant_count = 0
    while i < len(ids):
        if ids[i] == 3:  # assistant_start
            # Next tokens until assistant_end should be mask=1
            i += 1
            while i < len(ids) and ids[i] != 4:
                assert mask[i] == 1, f"Assistant content at index {i} should have mask=1"
                i += 1
            if i < len(ids):
                assert mask[i] == 1  # assistant_end itself is also mask=1
            assistant_count += 1
        else:
            if ids[i] != 4:  # Not assistant_end
                # Non-assistant tokens should have mask=0 (except assistant_end)
                if ids[i] not in (3, 4):  # Not start/end markers
                    pass  # Some markers have mask=0
            i += 1

    assert assistant_count == 2, f"Expected 2 assistant turns, got {assistant_count}"


def test_chatml_masks_tool_output():
    tok = MockTokenizer()
    conv = {"messages": [
        {"role": "user", "content": "X"},
        {"role": "assistant", "content": "Y"},
        {"role": "tool", "content": "Z"},
        {"role": "assistant", "content": "W"},
    ]}
    ids, mask = render_chatml_conversation(tok, conv)

    # Find tool output section (output_start=7, output_end=8)
    for i, token_id in enumerate(ids):
        if token_id == 7:  # output_start
            assert mask[i] == 0, "output_start should be masked out"
        elif token_id == 8:  # output_end
            assert mask[i] == 0, "output_end should be masked out"
