from typing import Union, Dict

def calculations_cost(total_tokens: int, input_tokens: int = None, output_tokens: int = None) -> Dict[str, Union[float, str]]:
    """
    Calculate the cost for OpenAI gpt-4.1-mini-2025-04-14 model usage based on tokens.
    
    Args:
        total_tokens (int): Total number of tokens used
        input_tokens (int, optional): Number of input tokens (if available)
        output_tokens (int, optional): Number of output tokens (if available)
        
    Returns:
        Dict containing cost breakdown and total cost
        
    Pricing for gpt-4.1-mini-2025-04-14 (as of 2024):
        - Input tokens: $5.00 per 1M tokens
        - Output tokens: $15.00 per 1M tokens
    """
    
    # GPT-4o pricing per 1M tokens (in USD)
    INPUT_COST_PER_1M = 5.00
    OUTPUT_COST_PER_1M = 15.00
    
    # If input/output tokens are provided separately
    if input_tokens is not None and output_tokens is not None:
        input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_1M
        output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
        total_cost = input_cost + output_cost
        
        return {
            "model": "gpt-4.1-mini-2025-04-14",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "cost_breakdown": f"Input: ${input_cost:.6f} + Output: ${output_cost:.6f} = ${total_cost:.6f}",
            "pricing_note": "Input: $5.00/1M tokens, Output: $15.00/1M tokens"
        }
    
    # If only total tokens provided, estimate based on typical ratio
    # Typical ratio is roughly 70% input, 30% output for most applications
    estimated_input_tokens = int(total_tokens * 0.7)
    estimated_output_tokens = int(total_tokens * 0.3)
    
    input_cost = (estimated_input_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (estimated_output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
    total_cost = input_cost + output_cost
    
    check =  {
        "model": "gpt-4.1-mini-2025-04-14",
        "total_tokens": total_tokens,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_input_cost_usd": round(input_cost, 6),
        "estimated_output_cost_usd": round(output_cost, 6),
        "estimated_total_cost_usd": round(total_cost, 6),
        "cost_breakdown": f"Estimated - Input: ${input_cost:.6f} + Output: ${output_cost:.6f} = ${total_cost:.6f}",
        "pricing_note": "Estimated based on 70% input / 30% output ratio",
        "warning": "For exact costs, provide input_tokens and output_tokens separately"
    }
    return {"estimated_total_cost_usd": round(total_cost, 6),}


def calculate_cost_from_usage(usage_object) -> Dict[str, Union[float, str]]:
    """
    Calculate cost from OpenAI usage object that contains prompt_tokens, completion_tokens, and total_tokens.
    
    Args:
        usage_object: OpenAI usage object with prompt_tokens, completion_tokens, total_tokens
        
    Returns:
        Dict containing detailed cost breakdown
    """
    
    if hasattr(usage_object, 'prompt_tokens') and hasattr(usage_object, 'completion_tokens'):
        return calculations_cost(
            total_tokens=usage_object.total_tokens,
            input_tokens=usage_object.prompt_tokens,
            output_tokens=usage_object.completion_tokens
        )
    else:
        return calculations_cost(total_tokens=usage_object.total_tokens)


def get_pricing_info() -> Dict[str, Union[str, float]]:
    """
    Get current pricing information for GPT-4o model.
    
    Returns:
        Dict containing pricing details
    """
    return {
        "model": "gpt-4.1-mini-2025-04-14",
        "input_cost_per_1m_tokens": 5.00,
        "output_cost_per_1m_tokens": 15.00,
        "currency": "USD",
        "last_updated": "2024",
        "note": "Pricing may change. Check OpenAI's official pricing page for latest rates."
    }


def calculate_budget_tokens(budget_usd: float, input_output_ratio: float = 0.7) -> Dict[str, int]:
    """
    Calculate how many tokens you can afford with a given budget.
    
    Args:
        budget_usd (float): Budget in USD
        input_output_ratio (float): Ratio of input tokens (default 0.7 = 70% input, 30% output)
        
    Returns:
        Dict containing token estimates for the budget
    """
    
    INPUT_COST_PER_1M = 5.00
    OUTPUT_COST_PER_1M = 15.00
    
    # Calculate weighted average cost per token
    input_ratio = input_output_ratio
    output_ratio = 1 - input_output_ratio
    
    weighted_cost_per_1m = (input_ratio * INPUT_COST_PER_1M) + (output_ratio * OUTPUT_COST_PER_1M)
    
    # Calculate total tokens possible
    total_tokens_possible = int((budget_usd / weighted_cost_per_1m) * 1_000_000)
    input_tokens_possible = int(total_tokens_possible * input_ratio)
    output_tokens_possible = int(total_tokens_possible * output_ratio)
    
    return {
        "budget_usd": budget_usd,
        "total_tokens_possible": total_tokens_possible,
        "input_tokens_possible": input_tokens_possible,
        "output_tokens_possible": output_tokens_possible,
        "input_output_ratio": f"{int(input_ratio*100)}% input / {int(output_ratio*100)}% output",
        "weighted_cost_per_1m_tokens": round(weighted_cost_per_1m, 2)
    }

