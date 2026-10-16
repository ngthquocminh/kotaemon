from kotaemon.llms import AzureChatOpenAI, ChatOpenAI


def _is_traced(client):
    # langsmith's wrap_openai patches `create` onto the instance
    return "create" in vars(client.chat.completions)


def test_openai_clients_are_wrapped():
    llm = ChatOpenAI(api_key="sk-test", model="gpt-4o-mini")
    assert _is_traced(llm.prepare_client())
    assert _is_traced(llm.prepare_client(async_version=True))
    azure = AzureChatOpenAI(
        azure_endpoint="https://x.openai.azure.com",
        api_key="k",
        api_version="2024-02-01",
        azure_deployment="d",
    )
    assert _is_traced(azure.prepare_client())
