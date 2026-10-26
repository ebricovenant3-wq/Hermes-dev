# Connect Hermes to OpenAI

Hermes already supports OpenAI API models. This connects Hermes to the OpenAI API; it does not sign into the ChatGPT website or provide access to ChatGPT conversations.

1. Create an API key in the [OpenAI API key settings](https://platform.openai.com/api-keys). Keep it private; do not paste it into Hermes chat.
2. In the private Hermes `env` file, set:

   ```env
   CEBRI_PROVIDER=openai
   CEBRI_MODEL=gpt-5-mini
   OPENAI_API_KEY=your-secret-key
   ```

   Replace the existing `CEBRI_PROVIDER` and `CEBRI_MODEL` values; do not add duplicate entries. The API key is stored server-side and the env file is owner-only.

3. Restart Hermes. Its `/status` response should report `provider: openai` and `mode: live` when the key and model initialize successfully.

OpenAI API usage has separate billing from ChatGPT subscriptions. See [OpenAI billing information](https://help.openai.com/en/articles/9039756-managing-billing-for-chatgpt-and-the-api-platform).
