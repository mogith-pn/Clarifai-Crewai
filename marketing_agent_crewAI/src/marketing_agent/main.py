#!/usr/bin/env python
import sys
from marketing_agent.crew import MarketingPostsCrew


def run(user_inputs=None):
    """
    Runs the marketing posts crew with provided inputs or fallbacks to default.
    Args:
        user_inputs (dict, optional): A dictionary of inputs for the crew. Defaults to None.
    """

    # define default inputs
    default_inputs = {
        "customer_domain": "nvidia.com/en-in/ai/",
        "project_description": """
Nvidia, a leading provider of NIMs, aims to revolutionize marketing automation for its enterprise clients. This project involves developing an innovative marketing strategy to showcase nvidia's NIMs, emphasizing ease of use, scalability, and integration capabilities. The campaign will target tech-savvy decision-makers in medium to large enterprises, highlighting success stories and the transformative potential of nvidia's platform.

Customer Domain: AI and Automation Solutions
Project Overview: Creating a comprehensive marketing campaign to boost awareness and adoption of nvidia's services among enterprise clients.
""",
    }
    
    # User user_inputs if provided, otherwise use default_inputs
    inputs = user_inputs if user_inputs is not None else default_inputs
    MarketingPostsCrew().crew().kickoff(inputs=inputs)


def train(user_inputs=None):
    """
    Trains the crew for a given number of iterations with provided inputs or fallbacks to default.
    Args:
        user_inputs (dict, optional): A dictionary of inputs for the crew. Defaults to None.
    """

    # define default inputs for training
    inputs = {
        "customer_domain": "nvidia.com/en-in/ai/",
        "project_description": """
Nvidia, a leading provider of gpus, aims to revolutionize marketing automation for its enterprise clients. This project involves developing an innovative marketing strategy to showcase nvidia's advanced gpu, emphasizing ease of use, scalability, and integration capabilities. The campaign will target tech-savvy decision-makers in medium to large enterprises, highlighting success stories and the transformative potential of nvidia's platform.

Customer Domain: AI and Automation Solutions
Project Overview: Creating a comprehensive marketing campaign to boost awareness and adoption of nvidia's services among enterprise clients.
""",
    }

    # Use user_inputs if provided, otherwise use default_inputs
    inputs = user_inputs if user_inputs is not None else default_inputs

    try:
        n_iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 1 # Default to 1 if no arg provided
        MarketingPostsCrew().crew().train(n_iterations=n_iterations, inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")