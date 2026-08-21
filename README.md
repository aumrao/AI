# AI

This repository contains multiple AI-based applications and experiments, organized into separate branches.

## Branches

### 1. moralquote

The `moralquote` branch contains an AI-powered Moral Quote application that automatically discovers, evaluates, and delivers an inspirational moral quote every morning.

### Application Workflow

The application follows this workflow:

1. **Quote Sources**
   - The application reads a list of URLs from a text file.
   - These URLs act as the configured sources for finding moral and inspirational quotes.

2. **Tavily Search**
   - Multiple AI agents use Tavily to retrieve relevant moral quotes from the configured sources.
   - The retrieved quotes are collected for further processing.

3. **Quote Selection using LLM**
   - The collected quotes are passed to an LLM.
   - The LLM evaluates the available quotes and selects the most meaningful and optimistic moral quote.

4. **Mobile Notification**
   - The selected quote is sent as a push notification to the mobile phone using the Pushover application.

5. **Email Notification**
   - The same selected quote is also sent to the configured email address.

### AWS Deployment Architecture

The application is containerized and deployed to AWS using the following workflow:

```text
Quote URLs
    |
    v
Tavily
    |
    v
Multiple AI Agents
    |
    v
Moral Quotes
    |
    v
LLM
    |
    v
Best Moral Quote
    |
    +------------------+
    |                  |
    v                  v
Pushover             Email
Mobile               Notification
Notification
