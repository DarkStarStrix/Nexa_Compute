"""GraphQL integration for inference server."""

from __future__ import annotations

from typing import Any

try:
    import graphene
except ImportError:  # pragma: no cover
    graphene = None  # type: ignore[assignment]


def build_graphql_app(resolver) -> Any:
    """Build a GraphQL app if graphene is available."""
    if graphene is None:
        raise ImportError("graphene is required for GraphQL endpoint. Install with `pip install graphene`.")

    class Query(graphene.ObjectType):
        health = graphene.String()

        def resolve_health(self, info):
            return "ok"

        embed = graphene.List(graphene.Float, mz=graphene.List(graphene.Float), intensity=graphene.List(graphene.Float))

        def resolve_embed(self, info, mz, intensity):
            return resolver(mz, intensity)

    schema = graphene.Schema(query=Query)
    from starlette.graphql import GraphQLApp  # type: ignore[attr-defined]

    return GraphQLApp(schema=schema)

