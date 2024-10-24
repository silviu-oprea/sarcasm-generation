from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ExplainableSarcasticResponse(object):
    """
    Output class for the sarcasm generation pipeline.
    """
    event: str
    failed_expectation: str
    relation_type: str
    relation_subject: str
    relation_object: str
    norm_violated: str
    response_texts: List[str]

    def to_json(self):
        return {
            "event": self.event,
            "failed_expectation": self.failed_expectation,
            "relation_type": self.relation_type,
            "relation_subject": self.relation_subject,
            "relation_object": self.relation_object,
            "norm_violated": self.norm_violated,
            "response_texts": self.response_texts
        }

@dataclass
class CommonsenseBuilderResponse(object):
    """
    Output class of the `build_commonsense` function called on objects that
    inherit from `max.commonsense_builders.builder.CommonsenseBuilder`.
    """
    event_obts: Dict[str, List[str]]
    failed_expectation_obts: Optional[Dict[str, List[str]]] = None
