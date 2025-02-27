from __future__ import annotations

import datetime
import json
import os
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, Self, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proxy_lite.environments import EnvironmentConfigTypes
from proxy_lite.environments.environment_base import Action, Observation
from proxy_lite.history import MessageHistory
from proxy_lite.solvers import SolverConfigTypes


class RunStatus(str, Enum):
    """Enum to track the status of a run."""

    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    FAILED = "failed"


class HistoryItem(BaseModel):
    """Base class for history items to allow for proper serialization/deserialization."""

    type: ClassVar[Literal["observation", "action"]]

    @classmethod
    def from_dict(cls, data: dict) -> Union[Observation, Action]:
        """Factory method to create the correct history item type."""
        if data.get("type") == "observation":
            return Observation(**data)
        elif data.get("type") == "action":
            return Action(**data)
        raise ValueError(f"Unknown history item type: {data.get('type')}")


class Run(BaseModel):
    """
    Represents a single run of a task through the system.

    Tracks all interactions between the environment and solver, along with
    metadata about the run's execution.
    """

    complete: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    task: str
    created_at: datetime.datetime
    status: RunStatus = RunStatus.INITIALIZING
    terminated_at: Optional[datetime.datetime] = None
    evaluation: dict[str, Any] | None = None
    history: list[Union[Observation, Action]] = Field(default_factory=list)
    solver_history: Optional[MessageHistory] = None
    result: str | None = None
    env_info: dict[str, Any] = Field(default_factory=dict)
    environment: Optional[EnvironmentConfigTypes] = None
    solver: Optional[SolverConfigTypes] = None

    # Path configuration
    STORAGE_DIR: ClassVar[str] = "local_trajectories"

    @classmethod
    def initialise(cls, task: str) -> Self:
        """Create a new run with a unique ID for the given task."""
        run_id = str(uuid.uuid4())
        return cls(
            run_id=run_id,
            task=task,
            created_at=datetime.datetime.now(datetime.UTC),
        )

    @classmethod
    def load(cls, run_id: str, base_path: Optional[Path] = None) -> Self:
        """Load a run from its JSON file."""
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent

        file_path = base_path / cls.STORAGE_DIR / f"{run_id}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Run file not found: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        # Convert string timestamps to datetime objects
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.datetime.fromisoformat(data["created_at"])
        if data.get("terminated_at") and isinstance(data["terminated_at"], str):
            data["terminated_at"] = datetime.datetime.fromisoformat(data["terminated_at"])

        # Convert history items to their proper types
        if "history" in data:
            data["history"] = [HistoryItem.from_dict(item) for item in data["history"]]

        return cls(**data)

    @property
    def observations(self) -> list[Observation]:
        """Get all observations in the history."""
        return [h for h in self.history if isinstance(h, Observation)]

    @property
    def actions(self) -> list[Action]:
        """Get all actions in the history."""
        return [h for h in self.history if isinstance(h, Action)]

    @property
    def last_action(self) -> Optional[Action]:
        """Get the most recent action, if any."""
        return self.actions[-1] if self.actions else None

    @property
    def last_observation(self) -> Optional[Observation]:
        """Get the most recent observation, if any."""
        return self.observations[-1] if self.observations else None

    @property
    def duration(self) -> Optional[datetime.timedelta]:
        """Calculate the duration of the run if it has terminated."""
        if self.terminated_at:
            return self.terminated_at - self.created_at
        return None

    def record(
        self,
        observation: Optional[Observation] = None,
        action: Optional[Action] = None,
        solver_history: Optional[MessageHistory] = None,
    ) -> None:
        """
        Record a new observation, action, or solver history in the run.

        Args:
            observation: An observation from the environment
            action: An action performed by the solver
            solver_history: The full message history from the solver

        Raises:
            ValueError: If both observation and action are provided
        """
        # Only one of observation and action should be provided to maintain proper ordering
        if observation and action:
            raise ValueError("Only one of observation and action can be provided")

        # Update run status if this is the first action or observation
        if (observation or action) and self.status == RunStatus.INITIALIZING:
            self.status = RunStatus.IN_PROGRESS

        if observation:
            self.history.append(observation)
        if action:
            self.history.append(action)
        if solver_history:
            self.solver_history = solver_history

    def mark_completed(self, result: str) -> None:
        """Mark the run as completed with a result."""
        self.result = result
        self.status = RunStatus.COMPLETED
        self.terminated_at = datetime.datetime.now(datetime.UTC)

    def terminate(self, failed: bool = False) -> None:
        """Terminate the run, optionally marking it as failed."""
        self.terminated_at = datetime.datetime.now(datetime.UTC)
        self.status = RunStatus.FAILED if failed else RunStatus.TERMINATED

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        """Validate that terminated_at is after created_at if both exist."""
        if self.terminated_at and self.created_at > self.terminated_at:
            raise ValueError("terminated_at must be after created_at")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert the run to a dictionary for serialization."""
        data = self.model_dump()
        # Convert datetime to ISO format strings for JSON serialization
        data["created_at"] = self.created_at.isoformat()
        if self.terminated_at:
            data["terminated_at"] = self.terminated_at.isoformat()
        return data


class DataRecorder:
    """
    Handles the persistence of run data to disk.

    Responsible for initializing, saving, and loading runs.
    """

    def __init__(self, storage_dir: Optional[str | Path] = None):
        """
        Initialize the data recorder.

        Args:
            storage_dir: Directory to store run data. If None, uses the default.
        """
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path(__file__).parent.parent.parent / Run.STORAGE_DIR

        # Ensure the storage directory exists
        os.makedirs(self.storage_dir, exist_ok=True)

    def initialise_run(self, task: str) -> Run:
        """Create a new run for the given task."""
        return Run.initialise(task)

    async def terminate(
        self,
        run: Run,
        save: bool = True,
        failed: bool = False,
    ) -> None:
        """
        Terminate a run and optionally save it.

        Args:
            run: The run to terminate
            save: Whether to save the run to disk
            failed: Whether the run failed
        """
        run.terminate(failed=failed)
        if save:
            await self.save(run)

    async def save(self, run: Run) -> None:
        """
        Save a run to disk.

        Args:
            run: The run to save
        """
        json_payload = run.to_dict()
        file_path = self.storage_dir / f"{run.run_id}.json"

        # Use atomic write pattern to prevent data corruption
        temp_path = file_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(json_payload, f, indent=2)

        # Atomic rename for safer file operations
        temp_path.rename(file_path)

    def load_run(self, run_id: str) -> Run:
        """
        Load a run from disk.

        Args:
            run_id: The ID of the run to load

        Returns:
            The loaded run

        Raises:
            FileNotFoundError: If the run file doesn't exist
        """
        return Run.load(run_id, base_path=self.storage_dir.parent)

    async def list_runs(self, limit: int = 100, status: Optional[RunStatus] = None) -> list[Run]:
        """
        List runs, optionally filtered by status.

        Args:
            limit: Maximum number of runs to return
            status: Filter runs by this status

        Returns:
            List of runs matching the criteria
        """
        runs = []
        for file_path in sorted(
            self.storage_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            if len(runs) >= limit:
                break

            try:
                run = Run.load(file_path.stem, base_path=self.storage_dir.parent)
                if status is None or run.status == status:
                    runs.append(run)
            except Exception as e:
                print(f"Error loading run {file_path}: {e}")

        return runs
