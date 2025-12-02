from textwrap import dedent

OBJECT_COMPARISON_PROMPT = dedent(
    """
<PROMPT>
    <ROLE>
        You are an expert semantic analyzer specializing in visual and functional object recognition. Your task is to match a detected object (candidate) with a reference object class.
    </ROLE>

    <INPUT_DATA>
        1. <PROCESSED_OBJECT>: The detected object and its description.
        2. <VIRTUAL_OBJECT>: The class name of the reference object.
    </INPUT_DATA>

    <INSTRUCTIONS>
        1. Determine if the PROCESSED_OBJECT represents the VIRTUAL_OBJECT, either directly or as its primary visual content.
        2. Output **True** if ANY of the following conditions are met:
            a. **Identity:** They are the same object, synonyms, or subclasses.
            b. **Primary Content / Purpose:** The VIRTUAL_OBJECT is designed specifically to hold or display the PROCESSED_OBJECT, and the PROCESSED_OBJECT is the main visual indicator of that function (e.g., detecting "clothes" validates "laundry basket"; detecting "books" validates "bookshelf"; detecting "towel" validates "towel rack").
            c. **Visual Occlusion:** The PROCESSED_OBJECT naturally covers the VIRTUAL_OBJECT in normal use, making them functionally equivalent for detection purposes.

        3. Output **False** if:
            a. **Incidental Proximity:** The objects are often found near each other but have distinct functions (e.g., "towel" is near "sink", but a sink is not a towel holder; "pillow" is near "nightstand", but distinct from it).
            b. **General Surface:** The VIRTUAL_OBJECT is a generic surface (like a table) and the PROCESSED_OBJECT is just an item on it, unless the item defines the object type (like "pool table").

        4. Output only one of the following (no explanations):
            - True
            - False
    </INSTRUCTIONS>
</PROMPT>
    """
)

BASE_PROMPT = dedent(
    """
<SYSTEM_ROLE>
You are an expert dataset generator for Semantic Navigation Evaluation. Your goal is to generate evaluation samples based strictly on the provided scene context.
</SYSTEM_ROLE>

<INPUT_CONTEXT>
The environment contains the following objects (format: - Object Type at Room):
{context}
</INPUT_CONTEXT>

<TASK>
{task}
</TASK>

<OUTPUT_FORMAT>
You must output a JSON object containing a list of samples.
- The output must be valid JSON.
- Do not include markdown code blocks (```json). Just the raw JSON string.
- Do not add explanations or conversational filler.
</OUTPUT_FORMAT>
"""
)


BASIC_QUESTION_PROMPT = dedent(
    """
<TASK_DEFINITION>
Generate {num_questions} "Explicit Retrieval" samples. These are direct commands asking to find a specific type of object.

<RULES>
1. **Target Existence:** The object type requested MUST exist in the context list.
2. **Class-Based:** The query should ask for an object class in specific (e.g., "Find a chair", "Where is the microwave?").
3. **Specifics:** You may add a location constraint if multiple objects of the same class exist (e.g., "The chair in the kitchen").
4. **Query Variety:** Use diverse phrasing for commands (e.g., "Take me to...", "Guide me to...", "I want to see...", "I want to go...").
5. **Robot Capabilities:** The robot can only respond to questions about its surroundings, escort users to locations, and guide them to objects.
6. **No Ambiguity:** Ensure the query is clear and unambiguous, avoiding vague terms that could refer to multiple objects.
7. **Remove IDs:** Do not include object and rooms IDs (_ID after the name) in the queries and responses.
8. **Single Object Guarantee:** Ensure that the query can be resolved to a single object in the context, either directly or through specified constraints. Do not create queries that could refer to multiple objects without further clarification.
</RULES>
</TASK_DEFINITION>

<JSON_SCHEMA>
{
  "samples": [
    {
      "query": "The user's direct command",
      "expected_answer": "What the robot should respond with (e.g., 'Taking you to the Chair in the Kitchen')",
    }
  ]
}
</JSON_SCHEMA>
"""
)

INDIRECT_QUESTION_PROMPT = dedent(
    """
<TASK_DEFINITION>
Generate {num_questions} "Implicit Query" samples based on affordances. These are statements of user intent without naming the object.

<RULES>
1. **Affordance Mapping:** The query expresses a need (e.g., "I am tired", "I need to throw this away").
2. **Valid Targets:** Identify ALL object classes in the context that logically satisfy this need.
3. **No ambiguity:** Ensure the query clearly implies a specific function or action that can be fulfilled by a single object in the context.
4. **No Direct Naming:** Do not mention the object name in the query.
5. **Anti-ambiguity Context:** You can provide context, like 'I'm in the kitchen' or 'The nearest one', to avoid multiple valid targets.
6. **Remove IDs:** Do not include object and rooms IDs (_ID after the name) in the queries and responses.
7. **Single Object Guarantee:** Ensure that the query can be resolved to a single object in the context, either directly or through specified constraints. Do not create queries that could refer to multiple objects without further clarification.
</RULES>
</TASK_DEFINITION>

<JSON_SCHEMA>
{
  "samples": [
    {
      "query": "User's statement of need (e.g., 'I need to wash my hands')",
      "expected_answer": "What the robot should respond with (e.g., 'Taking you to the Sink')"
    }
  ]
}
</JSON_SCHEMA>
"""
)

ADVERSARIAL_QUESTION_PROMPT = dedent(
    """
<TASK_DEFINITION>
Generate {num_questions} "Negative Constraint" samples. These are queries for object classes that are NOT present in the provided context list.

<RULES>
1. **Plausible Absence:** Focus on common household items that simply happen to be missing from this specific room/house list (e.g., searching for a 'Dishwasher' in a context that lists only a 'Fridge' and 'Oven').
2. **No Fabrication:** Do not invent fictional or highly unusual objects; stick to everyday items that could reasonably be expected in a household setting.
3. **Real Positioning:** Ensure the absent object is one that would logically fit within the environment described by the context (e.g., don't ask for a 'Boat' in a living room context).
4. **Single Object Guarantee:** Ensure that the query can be resolved to a single object in the context, either directly or through specified constraints. Do not create queries that could refer to multiple objects without further clarification.
</RULES>
</TASK_DEFINITION>

<JSON_SCHEMA>
{
  "samples": [
    {
      "query": "Request for a missing object class",
      "expected_answer": "What the robot should respond with (e.g., 'Sorry, there is no Dishwasher in this area.')"
    }
  ]
}
</JSON_SCHEMA>
"""
)

FOLLOW_UP_QUESTION_PROMPT = dedent(
    """
<TASK_DEFINITION>
Generate {num_questions} "Multi-turn Disambiguation" scenarios. These scenarios mimic an initial unclear request that needs clarification to specify the target class or location. The robot can only respond to questions about its surroundings, escort users to locations, and guide them to objects.

Rules:
1. **Ambiguity Trigger:** The `initial_query` must apply to at least TWO different object types OR the same object type in different rooms.
2. **Resolution:** The scenario must include the robot's question and the user's choice that resolves the ambiguity.
3. **Final Target:** The Ground Truth is the specific Class (and Room) chosen by the user in the final turn.
4. **Final Answer:** The robot's concluding statement confirming the target object and location.
5. **Robot Limitations:** The robot is only able to respond to questions about its surroundings, escort users to locations, and guide them to objects. The robot is not able to bring nothing.
6. **Grounded in Context:** Ensure all objects and rooms mentioned are part of the provided context.
7. **Context Enrichment:** You may add brief context to the initial query to enhance realism (e.g., "I'm in the kitchen", "The nearest one", "Any chair will do").
8. **Single Object Guarantee:** At the end of the interaction, ensure that the user's choice resolves to a single object in the context, either directly or through specified constraints. Do not create scenarios that could refer to multiple objects without further clarification.
</TASK_DEFINITION>

<JSON_SCHEMA>
{
  "samples": [
    {
      "messages": [
        {
          "role": "user",
          "content": "Initial ambiguous query from the user"
        },
        {
          "role": "robot",
          "content": "Clarifying question from the robot"
        },
        ...
        {
          "role": "user",
          "content": "User's choice to resolve ambiguity"
        }
      ],
    }
  ]
}
</JSON_SCHEMA>
"""
)

SEMANTIC_JUDGE_PROMPT = dedent(
    """
<ROLE>
    You are an expert judge evaluating the performance of a semantic navigation robot. Your task is to compare the "obtained response" against the "expected response" based on the User's Intent.
</ROLE>

<INPUT_DATA>
    Check the "is_follow_up" flag.
    Fields: "is_follow_up", "query", "expected_answer", "obtained_response", "messages", "obtained_messages".
</INPUT_DATA>

<ID_HANDLING_RULE>
    **CRITICAL:** Strip all numerical suffixes/IDs (e.g., "bathroom_1" -> "Bathroom"). Compare only semantic Classes.
</ID_HANDLING_RULE>

<EVALUATION_LOGIC>
    Analyze the input based on the 'is_follow_up' flag.

    ### CASE 1: Single Turn (is_follow_up = False)
    Compare 'obtained_response' with 'expected_answer' considering the 'query':
    1. **Intent Fulfillment:** Does the obtained object/action satisfy the user's underlying need stated in the query? (e.g., "Brushing teeth" requires a Toothbrush OR a Sink).
    2. **Functional Equivalence:** Even if the Object Class differs, is the obtained object a valid proxy for the task?
    3. **Location Viability:** Is the room type appropriate for the task, even if it's not the specific instance requested?

    ### CASE 2: Multi-Turn (is_follow_up = True)
    Compare flow and resolution. Did the robot resolve the ambiguity effectively?
</EVALUATION_LOGIC>

<SCORING_CRITERIA>
    Output one of the following verdicts:

    **True** (Strict Semantic Match):
    - The target Object Class AND Room Type match the expectation exactly (ignoring IDs).
    - The intent is perfectly captured.

    **Partial** (Functional Success / Valid Alternative):
    - **Functional Proxy:** The robot targets a different object class that logically supports the user's intent (e.g., User wants to "brush teeth", Robot goes to "Sink" instead of "Toothbrush". This is PARTIAL, not False).
    - **Location Alternative:** The robot finds a valid object/room for the task, even if it differs from the specific location expected (e.g., "Bathroom 2" instead of "Bathroom 1"), provided the task can be done there.
    - **Vagueness:** The robot identifies the correct object but omits the room name.

    **False** (Failure):
    - **Functional Mismatch:** The robot targets an object that has NO utility for the user's query (e.g., User wants to "brush teeth", Robot goes to "Bed").
    - **Hallucination:** Claims to find things that don't exist.
    - **Inaction:** Fails to guide or respond helpfully.
</SCORING_CRITERIA>

<OUTPUT_FORMAT>
    Output ONLY one single word: True, Partial, or False.
</OUTPUT_FORMAT>
    """
)
