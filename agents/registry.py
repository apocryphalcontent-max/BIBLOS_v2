"""
BIBLOS v2 - Agent Registry

Centralized registry for SDES extraction agents with dependency resolution
and lifecycle management.
"""
from typing import Dict, List, Optional, Type, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict

from agents.base import BaseExtractionAgent, AgentConfig, ExtractionType


@dataclass
class AgentRegistration:
    """Registration entry for an agent."""
    agent_class: Type[BaseExtractionAgent]
    config: AgentConfig
    instance: Optional[BaseExtractionAgent] = None
    dependencies: List[str] = field(default_factory=list)


class AgentRegistry:
    """
    Central registry for all SDES extraction agents.

    Provides:
    - Agent registration and discovery
    - Dependency resolution and topological ordering
    - Lifecycle management (init/shutdown)
    - Agent querying by type
    """

    _instance: Optional["AgentRegistry"] = None

    def __new__(cls) -> "AgentRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.logger = logging.getLogger("biblos.agents.registry")
        self._agents: Dict[str, AgentRegistration] = {}
        self._by_type: Dict[ExtractionType, List[str]] = defaultdict(list)
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._initialized = True

    def register(
        self,
        agent_class: Type[BaseExtractionAgent],
        config: AgentConfig
    ) -> None:
        """
        Register an agent class with its configuration.

        Args:
            agent_class: The agent class to register
            config: Configuration for the agent
        """
        if config.name in self._agents:
            self.logger.warning(f"Agent {config.name} already registered, overwriting")

        # Create instance to get dependencies
        instance = agent_class(config)
        dependencies = instance.get_dependencies()

        registration = AgentRegistration(
            agent_class=agent_class,
            config=config,
            instance=instance,
            dependencies=dependencies
        )

        self._agents[config.name] = registration
        self._by_type[config.extraction_type].append(config.name)
        self._dependency_graph[config.name] = set(dependencies)

        self.logger.info(f"Registered agent: {config.name} ({config.extraction_type.name})")

    def get(self, name: str) -> Optional[BaseExtractionAgent]:
        """Get agent instance by name."""
        registration = self._agents.get(name)
        return registration.instance if registration else None

    def get_by_type(self, extraction_type: ExtractionType) -> List[BaseExtractionAgent]:
        """Get all agents of a specific extraction type."""
        agents = []
        for name in self._by_type[extraction_type]:
            agent = self.get(name)
            if agent:
                agents.append(agent)
        return agents

    def get_all(self) -> List[BaseExtractionAgent]:
        """Get all registered agents."""
        return [reg.instance for reg in self._agents.values() if reg.instance]

    def get_execution_order(self) -> List[str]:
        """
        Get topologically sorted execution order respecting dependencies.

        Returns:
            List of agent names in dependency-respecting order
        """
        visited: Set[str] = set()
        order: List[str] = []
        temp_visited: Set[str] = set()

        def visit(name: str) -> None:
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return

            temp_visited.add(name)

            for dep in self._dependency_graph.get(name, set()):
                if dep in self._agents:
                    visit(dep)

            temp_visited.remove(name)
            visited.add(name)
            order.append(name)

        for name in self._agents:
            if name not in visited:
                visit(name)

        return order

    async def initialize_all(self) -> None:
        """Initialize all agents in dependency order."""
        order = self.get_execution_order()
        self.logger.info(f"Initializing {len(order)} agents")

        for name in order:
            agent = self.get(name)
            if agent:
                await agent.initialize()

    async def shutdown_all(self) -> None:
        """Shutdown all agents in reverse dependency order."""
        order = reversed(self.get_execution_order())

        for name in order:
            agent = self.get(name)
            if agent:
                await agent.shutdown()

    def list_agents(self) -> List[Dict]:
        """List all registered agents with their info."""
        return [
            {
                "name": name,
                "type": reg.config.extraction_type.name,
                "dependencies": reg.dependencies,
                "batch_size": reg.config.batch_size
            }
            for name, reg in self._agents.items()
        ]

    def clear(self) -> None:
        """Clear all registrations (for testing)."""
        self._agents.clear()
        self._by_type.clear()
        self._dependency_graph.clear()


# Global registry instance
registry = AgentRegistry()


def register_agent(config: AgentConfig):
    """Decorator for agent registration."""
    def decorator(cls: Type[BaseExtractionAgent]) -> Type[BaseExtractionAgent]:
        registry.register(cls, config)
        return cls
    return decorator
