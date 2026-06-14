"""relay — CloudCart's GenAI support agent.

Introduced in Module 2 of AWS GenAI Pro Mastery. This is the cumulative `relay/`
package the rest of the course builds on. By Module 3 it ships four pieces:

  - relay.models : the frozen Pydantic v2 schemas Ticket and Triage.
  - relay.config : the SOLE home of model-ID literals — the tier -> inference
                   profile map. Added in Module 3.
  - relay.llm    : converse() — the UNIQUE Bedrock call site (routing, streaming,
                   retries, cross-Region fallback). Added and FROZEN in Module 3;
                   its signature is byte-identical from M3 through M15.
  - relay.triage : raw ticket -> validated Triage JSON via Bedrock Prompt
                   Management. Module 3 refactored it to call
                   converse(tier="fast") instead of carrying its own model ID.
  - relay.kb     : retrieve() (Retrieve) + answer() (RetrieveAndGenerate) over the
                   managed Bedrock Knowledge Base `relay-kb` -> a grounded, cited
                   Answer. Added in Module 5. Introduces the Citation/Answer
                   schemas in relay.models, by addition.

Later modules extend this package by ADDITION only (relay.intake in Module 6,
relay.agent in Module 7, and so on). Nothing here is rewritten downstream.
"""

__all__ = ["models", "config", "llm", "triage", "kb"]
