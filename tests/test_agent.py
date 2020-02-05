import MixtureModelPong as mmp



def test_agent_init():
    agent = mmp.Agent()
    assert 3 == len(list(agent.parameters())[-1]), "Output dimension does not match action dimension"

if __name__ == '__main__':
    test_agent_init()