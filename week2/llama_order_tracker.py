import json
import requests
import gradio as gr
from ollama import chat  # Import the chat function directly

# System message for the assistant
system_message = """You are an AI assistant used internally at Pedigo Products by CSRs. You will be used primarily as a tool for interacting with Epicor.
You will always display returned information from tool responses in a human-readable format.
Whenever a User asks about an order, you should use the query_order_tracker tool to find relevant information.
If you return no results for a po_number, suggest that the user try the provided number as an order_number.
If you return no results for an order_number, suggest that the user try the provided number as po_number.
If the user provides a tracking number, use the order_number method."""


def query_order_tracker(po_number=None, order_number=None):
    """Queries the OrderTracker tool with the provided po_number or order_number."""
    print(
        f"Querying OrderTracker with po_number: {po_number}, order_number: {order_number}"
    )
    base_url = "http://localhost:5000"
    url = f"{base_url}/search"
    headers = {"Content-Type": "application/json"}
    data = {}
    if po_number:
        data["po_number"] = po_number
    if order_number:
        data["order_number"] = order_number

    try:
        print(f"Making API call to {url} with data: {data}")
        response = requests.post(url, headers=headers, json=data)
        print(f"Response Status: {response.status_code}, Response: {response.text}")
        response.raise_for_status()

        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        return {"error": str(e)}


def handle_tool_call(tool_name: str, arguments: dict):
    print(f"Handling tool call: {tool_name} with arguments: {arguments}")
    if tool_name in tools:
        result = tools[tool_name](**arguments)
        print(f"Tool Result: {result}")
        return result
    error_message = f"Tool {tool_name} not found"
    print(error_message)
    return {"error": error_message}


def format_tool_result(result):
    if not result.get("success"):
        return "No results found. Please try another query."

    contact_info = result["results"].get("contact_info", {})
    order_data = result["results"].get("order_data", [])

    formatted_result = "### Order Information:\n\n"

    # Add contact info
    formatted_result += f"**Contact Name**: {contact_info.get('name', 'N/A')}\n"
    formatted_result += f"**Contact Email**: {contact_info.get('email', 'N/A')}\n\n"

    # Add order details
    formatted_result += "**Order Details:**\n"
    for idx, order in enumerate(order_data, start=1):
        formatted_result += f"**Item {idx}:**\n"
        formatted_result += (
            f"- **Part Number**: {order.get('OrderDtl_PartNum', 'N/A')}\n"
        )
        formatted_result += (
            f"- **Description**: {order.get('OrderDtl_LineDesc', 'N/A')}\n"
        )
        formatted_result += f"- **Customer**: {order.get('Customer_Name', 'N/A')} ({order.get('Customer_CustID', 'N/A')})\n"
        formatted_result += (
            f"- **Order Number**: {order.get('OrderHed_OrderNum', 'N/A')}\n"
        )
        formatted_result += (
            f"- **Order Date**: {order.get('OrderHed_OrderDate', 'N/A')}\n"
        )
        formatted_result += (
            f"- **Required Date**: {order.get('OrderRel_ReqDate', 'N/A')}\n"
        )
        formatted_result += f"- **Carrier**: {order.get('CarrierNameDisplay', 'N/A')}\n"
        formatted_result += (
            f"- **Tracking Number**: {order.get('ShipHead_TrackingNumber', 'N/A')}\n"
        )
        formatted_result += f"- **Tracking URL**: [Track Package]({order.get('TrackingURL', 'N/A')})\n\n"

    return formatted_result


# Call Ollama
def call_ollama(history):
    # Prepare the conversation context
    messages = [{"role": "system", "content": system_message}] + history

    print("Sending the following messages to LLama:")
    for message in messages:
        print(message)

    # Generate response
    response = chat(
        model="llama3.2",
        messages=messages,
        tools=[query_order_tracker],
    )
    print(f"LLama Full Response: {response}")

    # Check for tool calls
    if response.message.tool_calls:
        print("Detected tool calls in LLama response.")
        for tool_call in response.message.tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments
            print(f"Processing tool call: {function_name} with arguments: {arguments}")
            tool_result = handle_tool_call(function_name, arguments)
            formatted_result = format_tool_result(tool_result)
            # Add formatted tool result to history
            history.append({"role": "assistant", "content": formatted_result})
    else:
        # Regular response
        output = response.message.content
        print(f"LLama Output: {output}")
        if output.strip():  # Add content only if it's not empty
            history.append({"role": "assistant", "content": output})
        else:
            print("No valid content or tool calls detected in the response.")

    return history


tools = {"query_order_tracker": query_order_tracker}


# Gradio UI
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=250, type="messages")
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
    with gr.Row():
        clear = gr.Button("Clear")

    def do_entry(message, history):
        history += [{"role": "user", "content": message}]
        return "", history

    def process_message(chatbot_history):
        chatbot_history = call_ollama(chatbot_history)
        return chatbot_history

    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        process_message, inputs=chatbot, outputs=chatbot
    )
    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

ui.launch(inbrowser=True)
