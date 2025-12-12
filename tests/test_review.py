#!/usr/bin/env python3
"""
Tests for Human Review System
=============================

Tests the review.py module functionality including:
- ReviewState dataclass (persistence, load/save)
- Approval and rejection workflows
- Spec change detection (hash validation)
- Display functions
- Review status summary
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from review import (
    ReviewState,
    ReviewChoice,
    REVIEW_STATE_FILE,
    _compute_file_hash,
    _compute_spec_hash,
    _extract_section,
    _truncate_text,
    get_review_status_summary,
    get_review_menu_options,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def review_spec_dir(tmp_path: Path) -> Path:
    """Create a spec directory with spec.md and implementation_plan.json."""
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir(parents=True)

    # Create spec.md
    spec_content = """# Test Feature

## Overview

This is a test feature specification for unit testing purposes.

## Workflow Type

**Type**: feature

## Files to Modify

| File | Service | What to Change |
|------|---------|---------------|
| `app/main.py` | backend | Add new endpoint |
| `src/components/Test.tsx` | frontend | Add new component |

## Files to Create

| File | Service | Purpose |
|------|---------|---------|
| `app/utils/helper.py` | backend | Helper functions |

## Success Criteria

The task is complete when:

- [ ] New endpoint responds correctly
- [ ] Component renders without errors
- [ ] All tests pass
"""
    (spec_dir / "spec.md").write_text(spec_content)

    # Create implementation_plan.json
    plan = {
        "feature": "Test Feature",
        "workflow_type": "feature",
        "services_involved": ["backend", "frontend"],
        "phases": [
            {
                "phase": 1,
                "name": "Backend Foundation",
                "type": "setup",
                "chunks": [
                    {
                        "id": "chunk-1-1",
                        "description": "Add new endpoint",
                        "service": "backend",
                        "status": "pending",
                    },
                ],
            },
        ],
        "final_acceptance": ["Feature works correctly"],
        "summary": {
            "total_phases": 1,
            "total_chunks": 1,
        },
    }
    (spec_dir / "implementation_plan.json").write_text(json.dumps(plan, indent=2))

    return spec_dir


@pytest.fixture
def approved_state() -> ReviewState:
    """Create an approved ReviewState."""
    return ReviewState(
        approved=True,
        approved_by="test_user",
        approved_at="2024-01-15T10:30:00",
        feedback=["Looks good!", "Minor suggestion added."],
        spec_hash="abc123",
        review_count=2,
    )


@pytest.fixture
def pending_state() -> ReviewState:
    """Create a pending (not approved) ReviewState."""
    return ReviewState(
        approved=False,
        approved_by="",
        approved_at="",
        feedback=["Need more details on API."],
        spec_hash="",
        review_count=1,
    )


# =============================================================================
# REVIEW STATE - BASIC FUNCTIONALITY
# =============================================================================

class TestReviewStateBasics:
    """Tests for ReviewState basic functionality."""

    def test_default_state(self):
        """New ReviewState has correct defaults."""
        state = ReviewState()

        assert state.approved is False
        assert state.approved_by == ""
        assert state.approved_at == ""
        assert state.feedback == []
        assert state.spec_hash == ""
        assert state.review_count == 0

    def test_to_dict(self, approved_state: ReviewState):
        """to_dict() returns correct dictionary."""
        d = approved_state.to_dict()

        assert d["approved"] is True
        assert d["approved_by"] == "test_user"
        assert d["approved_at"] == "2024-01-15T10:30:00"
        assert d["feedback"] == ["Looks good!", "Minor suggestion added."]
        assert d["spec_hash"] == "abc123"
        assert d["review_count"] == 2

    def test_from_dict(self):
        """from_dict() creates correct ReviewState."""
        data = {
            "approved": True,
            "approved_by": "user1",
            "approved_at": "2024-02-20T14:00:00",
            "feedback": ["Test feedback"],
            "spec_hash": "xyz789",
            "review_count": 5,
        }

        state = ReviewState.from_dict(data)

        assert state.approved is True
        assert state.approved_by == "user1"
        assert state.approved_at == "2024-02-20T14:00:00"
        assert state.feedback == ["Test feedback"]
        assert state.spec_hash == "xyz789"
        assert state.review_count == 5

    def test_from_dict_with_missing_fields(self):
        """from_dict() handles missing fields with defaults."""
        data = {"approved": True}

        state = ReviewState.from_dict(data)

        assert state.approved is True
        assert state.approved_by == ""
        assert state.approved_at == ""
        assert state.feedback == []
        assert state.spec_hash == ""
        assert state.review_count == 0

    def test_from_dict_empty(self):
        """from_dict() handles empty dictionary."""
        state = ReviewState.from_dict({})

        assert state.approved is False
        assert state.approved_by == ""
        assert state.review_count == 0


# =============================================================================
# REVIEW STATE - LOAD/SAVE
# =============================================================================

class TestReviewStatePersistence:
    """Tests for ReviewState load and save operations."""

    def test_save_creates_file(self, tmp_path: Path):
        """save() creates review_state.json file."""
        spec_dir = tmp_path / "spec"
        spec_dir.mkdir()

        state = ReviewState(approved=True, approved_by="user")
        state.save(spec_dir)

        state_file = spec_dir / REVIEW_STATE_FILE
        assert state_file.exists()

    def test_save_writes_correct_json(self, tmp_path: Path):
        """save() writes correct JSON content."""
        spec_dir = tmp_path / "spec"
        spec_dir.mkdir()

        state = ReviewState(
            approved=True,
            approved_by="test_user",
            approved_at="2024-01-01T00:00:00",
            feedback=["Good work"],
            spec_hash="hash123",
            review_count=3,
        )
        state.save(spec_dir)

        state_file = spec_dir / REVIEW_STATE_FILE
        with open(state_file) as f:
            data = json.load(f)

        assert data["approved"] is True
        assert data["approved_by"] == "test_user"
        assert data["feedback"] == ["Good work"]
        assert data["review_count"] == 3

    def test_load_existing_file(self, tmp_path: Path):
        """load() reads existing review_state.json file."""
        spec_dir = tmp_path / "spec"
        spec_dir.mkdir()

        # Create state file manually
        data = {
            "approved": True,
            "approved_by": "manual_user",
            "approved_at": "2024-03-15T09:00:00",
            "feedback": ["Manually created"],
            "spec_hash": "manual_hash",
            "review_count": 1,
        }
        state_file = spec_dir / REVIEW_STATE_FILE
        state_file.write_text(json.dumps(data))

        state = ReviewState.load(spec_dir)

        assert state.approved is True
        assert state.approved_by == "manual_user"
        assert state.feedback == ["Manually created"]

    def test_load_missing_file(self, tmp_path: Path):
        """load() returns empty state when file doesn't exist."""
        spec_dir = tmp_path / "spec"
        spec_dir.mkdir()

        state = ReviewState.load(spec_dir)

        assert state.approved is False
        assert state.approved_by == ""
        assert state.review_count == 0

    def test_load_corrupted_json(self, tmp_path: Path):
        """load() returns empty state for corrupted JSON."""
        spec_dir = tmp_path / "spec"
        spec_dir.mkdir()

        state_file = spec_dir / REVIEW_STATE_FILE
        state_file.write_text("{ invalid json }")

        state = ReviewState.load(spec_dir)

        assert state.approved is False
        assert state.review_count == 0

    def test_load_empty_file(self, tmp_path: Path):
        """load() returns empty state for empty file."""
        spec_dir = tmp_path / "spec"
        spec_dir.mkdir()

        state_file = spec_dir / REVIEW_STATE_FILE
        state_file.write_text("")

        state = ReviewState.load(spec_dir)

        assert state.approved is False

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """save() and load() preserve state correctly."""
        spec_dir = tmp_path / "spec"
        spec_dir.mkdir()

        original = ReviewState(
            approved=True,
            approved_by="roundtrip_user",
            approved_at="2024-06-01T12:00:00",
            feedback=["First review", "Second review"],
            spec_hash="roundtrip_hash",
            review_count=7,
        )
        original.save(spec_dir)

        loaded = ReviewState.load(spec_dir)

        assert loaded.approved == original.approved
        assert loaded.approved_by == original.approved_by
        assert loaded.approved_at == original.approved_at
        assert loaded.feedback == original.feedback
        assert loaded.spec_hash == original.spec_hash
        assert loaded.review_count == original.review_count


# =============================================================================
# REVIEW STATE - APPROVAL METHODS
# =============================================================================

class TestReviewStateApproval:
    """Tests for approve(), reject(), and related methods."""

    def test_is_approved_true(self, approved_state: ReviewState):
        """is_approved() returns True for approved state."""
        assert approved_state.is_approved() is True

    def test_is_approved_false(self, pending_state: ReviewState):
        """is_approved() returns False for pending state."""
        assert pending_state.is_approved() is False

    def test_approve_sets_fields(self, review_spec_dir: Path):
        """approve() sets all required fields correctly."""
        state = ReviewState()

        # Freeze time for consistent testing
        with patch("review.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2024-07-01T10:00:00"
            state.approve(review_spec_dir, approved_by="approver")

        assert state.approved is True
        assert state.approved_by == "approver"
        assert state.approved_at == "2024-07-01T10:00:00"
        assert state.spec_hash != ""  # Hash should be computed
        assert state.review_count == 1

    def test_approve_increments_review_count(self, review_spec_dir: Path):
        """approve() increments review_count each time."""
        state = ReviewState(review_count=3)

        state.approve(review_spec_dir, approved_by="user", auto_save=False)

        assert state.review_count == 4

    def test_approve_auto_saves(self, review_spec_dir: Path):
        """approve() saves state when auto_save=True (default)."""
        state = ReviewState()
        state.approve(review_spec_dir, approved_by="user")

        state_file = review_spec_dir / REVIEW_STATE_FILE
        assert state_file.exists()

        loaded = ReviewState.load(review_spec_dir)
        assert loaded.approved is True

    def test_approve_no_auto_save(self, review_spec_dir: Path):
        """approve() doesn't save when auto_save=False."""
        state = ReviewState()
        state.approve(review_spec_dir, approved_by="user", auto_save=False)

        state_file = review_spec_dir / REVIEW_STATE_FILE
        assert not state_file.exists()

    def test_reject_clears_approval(self, review_spec_dir: Path):
        """reject() clears approval fields."""
        state = ReviewState(
            approved=True,
            approved_by="old_user",
            approved_at="2024-01-01T00:00:00",
            spec_hash="old_hash",
            review_count=5,
        )

        state.reject(review_spec_dir, auto_save=False)

        assert state.approved is False
        assert state.approved_by == ""
        assert state.approved_at == ""
        assert state.spec_hash == ""
        assert state.review_count == 6  # Still incremented

    def test_invalidate_keeps_feedback(self, review_spec_dir: Path):
        """invalidate() keeps feedback history."""
        state = ReviewState(
            approved=True,
            approved_by="user",
            feedback=["Important feedback"],
            spec_hash="hash",
        )

        state.invalidate(review_spec_dir, auto_save=False)

        assert state.approved is False
        assert state.spec_hash == ""
        assert state.feedback == ["Important feedback"]  # Preserved
        assert state.approved_by == "user"  # Kept as history


# =============================================================================
# REVIEW STATE - HASH VALIDATION
# =============================================================================

class TestSpecHashValidation:
    """Tests for spec change detection using hash."""

    def test_compute_file_hash_existing_file(self, tmp_path: Path):
        """_compute_file_hash() returns hash for existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        file_hash = _compute_file_hash(test_file)

        # Verify it's a valid MD5 hash
        assert len(file_hash) == 32
        assert all(c in "0123456789abcdef" for c in file_hash)

    def test_compute_file_hash_missing_file(self, tmp_path: Path):
        """_compute_file_hash() returns empty string for missing file."""
        missing_file = tmp_path / "nonexistent.txt"

        file_hash = _compute_file_hash(missing_file)

        assert file_hash == ""

    def test_compute_file_hash_deterministic(self, tmp_path: Path):
        """_compute_file_hash() returns same hash for same content."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Consistent content")

        hash1 = _compute_file_hash(test_file)
        hash2 = _compute_file_hash(test_file)

        assert hash1 == hash2

    def test_compute_file_hash_different_content(self, tmp_path: Path):
        """_compute_file_hash() returns different hash for different content."""
        test_file = tmp_path / "test.txt"

        test_file.write_text("Content A")
        hash_a = _compute_file_hash(test_file)

        test_file.write_text("Content B")
        hash_b = _compute_file_hash(test_file)

        assert hash_a != hash_b

    def test_compute_spec_hash(self, review_spec_dir: Path):
        """_compute_spec_hash() computes combined hash of spec files."""
        spec_hash = _compute_spec_hash(review_spec_dir)

        # Should be a valid MD5 hash
        assert len(spec_hash) == 32
        assert all(c in "0123456789abcdef" for c in spec_hash)

    def test_compute_spec_hash_changes_on_spec_edit(self, review_spec_dir: Path):
        """_compute_spec_hash() changes when spec.md is modified."""
        hash_before = _compute_spec_hash(review_spec_dir)

        # Modify spec.md
        spec_file = review_spec_dir / "spec.md"
        spec_file.write_text("Modified content")

        hash_after = _compute_spec_hash(review_spec_dir)

        assert hash_before != hash_after

    def test_compute_spec_hash_changes_on_plan_edit(self, review_spec_dir: Path):
        """_compute_spec_hash() changes when plan is modified."""
        hash_before = _compute_spec_hash(review_spec_dir)

        # Modify implementation_plan.json
        plan_file = review_spec_dir / "implementation_plan.json"
        plan_file.write_text('{"modified": true}')

        hash_after = _compute_spec_hash(review_spec_dir)

        assert hash_before != hash_after

    def test_is_approval_valid_with_matching_hash(self, review_spec_dir: Path):
        """is_approval_valid() returns True when hash matches."""
        state = ReviewState()
        state.approve(review_spec_dir, approved_by="user", auto_save=False)

        assert state.is_approval_valid(review_spec_dir) is True

    def test_is_approval_valid_with_changed_spec(self, review_spec_dir: Path):
        """is_approval_valid() returns False when spec changed."""
        state = ReviewState()
        state.approve(review_spec_dir, approved_by="user", auto_save=False)

        # Modify spec after approval
        spec_file = review_spec_dir / "spec.md"
        spec_file.write_text("New content after approval")

        assert state.is_approval_valid(review_spec_dir) is False

    def test_is_approval_valid_not_approved(self, review_spec_dir: Path):
        """is_approval_valid() returns False when not approved."""
        state = ReviewState(approved=False)

        assert state.is_approval_valid(review_spec_dir) is False

    def test_is_approval_valid_legacy_no_hash(self, review_spec_dir: Path):
        """is_approval_valid() returns True for legacy approvals without hash."""
        state = ReviewState(
            approved=True,
            approved_by="legacy_user",
            spec_hash="",  # No hash (legacy approval)
        )

        assert state.is_approval_valid(review_spec_dir) is True


# =============================================================================
# REVIEW STATE - FEEDBACK
# =============================================================================

class TestReviewStateFeedback:
    """Tests for feedback functionality."""

    def test_add_feedback(self, tmp_path: Path):
        """add_feedback() adds timestamped feedback."""
        spec_dir = tmp_path / "spec"
        spec_dir.mkdir()

        state = ReviewState()
        state.add_feedback("Great work!", spec_dir, auto_save=False)

        assert len(state.feedback) == 1
        # Should have timestamp prefix
        assert "]" in state.feedback[0]
        assert "Great work!" in state.feedback[0]

    def test_add_multiple_feedback(self, tmp_path: Path):
        """add_feedback() accumulates feedback."""
        spec_dir = tmp_path / "spec"
        spec_dir.mkdir()

        state = ReviewState()
        state.add_feedback("First comment", spec_dir, auto_save=False)
        state.add_feedback("Second comment", spec_dir, auto_save=False)

        assert len(state.feedback) == 2
        assert "First comment" in state.feedback[0]
        assert "Second comment" in state.feedback[1]

    def test_add_feedback_auto_saves(self, tmp_path: Path):
        """add_feedback() saves when auto_save=True."""
        spec_dir = tmp_path / "spec"
        spec_dir.mkdir()

        state = ReviewState()
        state.add_feedback("Saved feedback", spec_dir, auto_save=True)

        loaded = ReviewState.load(spec_dir)
        assert len(loaded.feedback) == 1
        assert "Saved feedback" in loaded.feedback[0]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_extract_section_found(self):
        """_extract_section() extracts content correctly."""
        content = """# Title

## Overview

This is the overview section.

## Details

This is the details section.
"""
        overview = _extract_section(content, "## Overview")

        assert "This is the overview section." in overview
        assert "This is the details section." not in overview

    def test_extract_section_not_found(self):
        """_extract_section() returns empty string when not found."""
        content = """# Title

## Existing Section

Content here.
"""
        result = _extract_section(content, "## Missing Section")

        assert result == ""

    def test_extract_section_last_section(self):
        """_extract_section() handles last section correctly."""
        content = """# Title

## First

First content.

## Last

Last content.
"""
        last = _extract_section(content, "## Last")

        assert "Last content." in last

    def test_truncate_text_short(self):
        """_truncate_text() returns short text unchanged."""
        short_text = "Short text"

        result = _truncate_text(short_text, max_lines=10, max_chars=100)

        assert result == "Short text"

    def test_truncate_text_too_many_lines(self):
        """_truncate_text() truncates by line count."""
        long_text = "\n".join(f"Line {i}" for i in range(20))

        result = _truncate_text(long_text, max_lines=5, max_chars=1000)

        # Should contain 5 lines from original + "..." on new line
        lines = result.split("\n")
        assert lines[-1] == "..."
        assert len(lines) <= 6  # 5 content lines + "..." line
        assert "Line 0" in result
        assert "Line 4" in result

    def test_truncate_text_too_many_chars(self):
        """_truncate_text() truncates by character count."""
        long_text = "A" * 500

        result = _truncate_text(long_text, max_lines=100, max_chars=100)

        assert len(result) <= 100
        assert result.endswith("...")


# =============================================================================
# REVIEW STATUS SUMMARY
# =============================================================================

class TestReviewStatusSummary:
    """Tests for get_review_status_summary()."""

    def test_summary_approved_valid(self, review_spec_dir: Path):
        """Summary for approved and valid state."""
        state = ReviewState()
        state.approve(review_spec_dir, approved_by="summary_user")

        summary = get_review_status_summary(review_spec_dir)

        assert summary["approved"] is True
        assert summary["valid"] is True
        assert summary["approved_by"] == "summary_user"
        assert summary["spec_changed"] is False

    def test_summary_approved_stale(self, review_spec_dir: Path):
        """Summary for approved but stale state."""
        state = ReviewState()
        state.approve(review_spec_dir, approved_by="user")

        # Modify spec after approval
        (review_spec_dir / "spec.md").write_text("Changed!")

        summary = get_review_status_summary(review_spec_dir)

        assert summary["approved"] is True
        assert summary["valid"] is False
        assert summary["spec_changed"] is True

    def test_summary_not_approved(self, review_spec_dir: Path):
        """Summary for not approved state."""
        summary = get_review_status_summary(review_spec_dir)

        assert summary["approved"] is False
        assert summary["valid"] is False
        assert summary["approved_by"] == ""

    def test_summary_with_feedback(self, review_spec_dir: Path):
        """Summary includes feedback count."""
        state = ReviewState(feedback=["One", "Two", "Three"])
        state.save(review_spec_dir)

        summary = get_review_status_summary(review_spec_dir)

        assert summary["feedback_count"] == 3


# =============================================================================
# REVIEW MENU OPTIONS
# =============================================================================

class TestReviewMenuOptions:
    """Tests for review menu configuration."""

    def test_get_review_menu_options_count(self):
        """get_review_menu_options() returns correct number of options."""
        options = get_review_menu_options()

        assert len(options) == 5

    def test_get_review_menu_options_keys(self):
        """get_review_menu_options() has correct keys."""
        options = get_review_menu_options()
        keys = [opt.key for opt in options]

        assert ReviewChoice.APPROVE.value in keys
        assert ReviewChoice.EDIT_SPEC.value in keys
        assert ReviewChoice.EDIT_PLAN.value in keys
        assert ReviewChoice.FEEDBACK.value in keys
        assert ReviewChoice.REJECT.value in keys

    def test_get_review_menu_options_have_labels(self):
        """All menu options have labels and descriptions."""
        options = get_review_menu_options()

        for opt in options:
            assert opt.label != ""
            assert opt.description != ""

    def test_review_choice_enum_values(self):
        """ReviewChoice enum has expected values."""
        assert ReviewChoice.APPROVE.value == "approve"
        assert ReviewChoice.EDIT_SPEC.value == "edit_spec"
        assert ReviewChoice.EDIT_PLAN.value == "edit_plan"
        assert ReviewChoice.FEEDBACK.value == "feedback"
        assert ReviewChoice.REJECT.value == "reject"


# =============================================================================
# FULL REVIEW FLOW (INTEGRATION)
# =============================================================================

class TestFullReviewFlow:
    """Integration tests for full review workflow."""

    def test_full_approval_flow(self, review_spec_dir: Path):
        """Test complete approval flow."""
        # 1. Initially not approved
        state = ReviewState.load(review_spec_dir)
        assert not state.is_approved()

        # 2. Add feedback
        state.add_feedback("Needs minor changes", review_spec_dir)

        # 3. Approve
        state.approve(review_spec_dir, approved_by="reviewer")

        # 4. Verify state
        assert state.is_approved()
        assert state.is_approval_valid(review_spec_dir)

        # 5. Reload and verify persisted
        reloaded = ReviewState.load(review_spec_dir)
        assert reloaded.is_approved()
        assert reloaded.approved_by == "reviewer"
        assert len(reloaded.feedback) == 1

    def test_approval_invalidation_on_change(self, review_spec_dir: Path):
        """Test that spec changes invalidate approval."""
        # 1. Approve initially
        state = ReviewState()
        state.approve(review_spec_dir, approved_by="user")
        assert state.is_approval_valid(review_spec_dir)

        # 2. Modify spec.md
        spec_file = review_spec_dir / "spec.md"
        original_content = spec_file.read_text()
        spec_file.write_text(original_content + "\n## New Section\n\nAdded content.")

        # 3. Approval should now be invalid
        assert not state.is_approval_valid(review_spec_dir)

        # 4. Re-approve with new hash
        state.approve(review_spec_dir, approved_by="user")
        assert state.is_approval_valid(review_spec_dir)

    def test_rejection_flow(self, review_spec_dir: Path):
        """Test rejection workflow."""
        # 1. Approve first
        state = ReviewState()
        state.approve(review_spec_dir, approved_by="user")
        assert state.is_approved()

        # 2. Reject
        state.reject(review_spec_dir)

        # 3. Verify state
        assert not state.is_approved()

        # 4. Reload and verify
        reloaded = ReviewState.load(review_spec_dir)
        assert not reloaded.is_approved()

    def test_auto_approve_flow(self, review_spec_dir: Path):
        """Test auto-approve workflow."""
        state = ReviewState()
        state.approve(review_spec_dir, approved_by="auto")

        assert state.is_approved()
        assert state.approved_by == "auto"
        assert state.is_approval_valid(review_spec_dir)

    def test_multiple_review_sessions(self, review_spec_dir: Path):
        """Test multiple review sessions increment count correctly."""
        state = ReviewState()
        assert state.review_count == 0

        # First review - approve
        state.approve(review_spec_dir, approved_by="user1")
        assert state.review_count == 1

        # Modify spec to invalidate
        (review_spec_dir / "spec.md").write_text("Changed content")
        state.invalidate(review_spec_dir)

        # Second review - reject
        state.reject(review_spec_dir)
        assert state.review_count == 2

        # Third review - approve again
        state.approve(review_spec_dir, approved_by="user2")
        assert state.review_count == 3
