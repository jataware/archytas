from archytas.models.base import ModelConfig
from archytas.models.bedrock import BedrockModel

def run_bedrock_interactive():
    from archytas.react import ReActAgent
    from easyrepl import REPL

    agent = ReActAgent(
        model=BedrockModel(
            credentials_profile_name='default',
            config=ModelConfig(
                model_name='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
                region='us-east-1'
            )
        ), 
        verbose=True
    )
    print(f'prompt:\n```\n{agent.prompt}\n```')

    for query in REPL(history_file='.history'):
        response = agent.react(query)
        print(response)

if __name__ == '__main__':
    run_bedrock_interactive()
