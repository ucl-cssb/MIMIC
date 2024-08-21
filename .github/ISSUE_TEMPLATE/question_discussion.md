---
name: Question or Discussion
about: Ask a question or start a discussion
title: "[QUESTION/DISCUSSION]"
labels: question/discussion
assignees: ''

---

body:
- type: markdown
  attributes:
    value: "## Question or Discussion\nPlease fill out the following sections to start a discussion or ask a question."

- type: input
  id: question-title
  attributes:
    label: Question/Discussion Title
    description: "Provide a brief summary of your question or discussion topic."
    placeholder: "Brief summary"
  validations:
    required: true

- type: textarea
  id: question-description
  attributes:
    label: Describe your question or topic
    description: "A clear and concise description of your question or discussion topic."
    placeholder: "Describe the question or topic in detail"
  validations:
    required: true

- type: input
  id: question-context
  attributes:
    label: Context
    description: "Provide any background information or context that is relevant."
    placeholder: "Provide the context"

- type: textarea
  id: expected-outcome
  attributes:
    label: Expected Outcome
    description: "What you hope to get out of the discussion (e.g., decision, clarification, etc.)."
    placeholder: "Describe the expected outcome"

- type: textarea
  id: additional-context
  attributes:
    label: Additional Context
    description: "Add any other context or screenshots relevant to the discussion here."
    placeholder: "Any other details or attachments?"

- type: checkboxes
  id: optional-labels
  attributes:
    label: Optional Labels
    description: "If any of these optional labels are necessary, please select them."
    options:
      - label: HIGH priority
      - label: LOW priority
      - label: good first issue
      - label: help wanted
      - label: wontfix
