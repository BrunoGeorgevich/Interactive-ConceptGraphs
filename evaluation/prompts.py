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

<QUESTION_RULES>
 - Is totally prohibited to ask for objects that are not in the provided context, except when generating "Negative Constraint" samples.
 - It is mandatory to always ensure that the generated query can be resolved to a single object in the context, either directly or through specified constraints. Do not create queries that could refer to multiple objects without further clarification.
 - The robot always go to the object or location when asked, never bring the object to the user or any other action.
 - Do not generate "Where questions" or "Yes/No questions".
</QUESTION_RULES>

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
9. **IDs Prohibition:** Considering that the ground truth and processing was performed in different mapping, the object and rooms IDs are different. Because of that, do not include object and rooms IDs (_ID after the name) in the queries and responses. Also, do not rely on IDs to differentiate objects or rooms in the query. It is TOTALLY PROHIBITED to use IDs in the queries and responses.
10. **Room or Object Specification:** Do not specify which room or object with a number (as with the ID or in full), for instance "bedroom 2" or "bed_3". Always use only the class or room name (e.g., "bedroom" or "bed").
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
8. **IDs Prohibition:** Considering that the ground truth and processing was performed in different mapping, the object and rooms IDs are different. Because of that, do not include object and rooms IDs (_ID after the name) in the queries and responses. Also, do not rely on IDs to differentiate objects or rooms in the query. It is TOTALLY PROHIBITED to use IDs in the queries and responses.
9. **Room or Object Specification:** Do not specify which room or object with a number (as with the ID or in full), for instance "bedroom 2" or "bed_3". Always use only the class or room name (e.g., "bedroom" or "bed").
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
1. **Plausible Absence:** Focus on common household items that simply happen to be missing from this specific room/house list (e.g., searching for a 'Dishwasher' in a context that lists only a 'Fridge' and 'Oven'). Do not use objects with names different from existing ones if they are essentially the same (e.g., do not ask for 'Sofa' if 'Couch' is already present). Instead, use objects that are guaranteed not to be in the current context.
2. **No Fabrication:** Do not invent fictional or highly unusual objects; stick to everyday items that could reasonably be expected in a household setting.
3. **Real Positioning:** Ensure the absent object is one that would logically fit within the environment described by the context (e.g., don't ask for a 'Boat' in a living room context).
4. **Single Object Guarantee:** Ensure that the query can be resolved to a single object in the context, either directly or through specified constraints. Do not create queries that could refer to multiple objects without further clarification.
5. **IDs Prohibition:** Considering that the ground truth and processing was performed in different mapping, the object and rooms IDs are different. Because of that, do not include object and rooms IDs (_ID after the name) in the queries and responses. Also, do not rely on IDs to differentiate objects or rooms in the query. It is TOTALLY PROHIBITED to use IDs in the queries and responses.
6. **Room or Object Specification:** Do not specify which room or object with a number (as with the ID or in full), for instance "bedroom 2" or "bed_3". Always use only the class or room name (e.g., "bedroom" or "bed").
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
9. **IDs Mismatching:** Considering that the ground truth and processing was performed in different mapping, the object and rooms IDs are different. Because of that, do not include object and rooms IDs (_ID after the name) in the queries and responses. Also, do not rely on IDs to differentiate objects or rooms in the disambiguation process.
10. **IDs Prohibition:** Considering that the ground truth and processing was performed in different mapping, the object and rooms IDs are different. Because of that, do not include object and rooms IDs (_ID after the name) in the queries and responses. Also, do not rely on IDs to differentiate objects or rooms in the query. It is TOTALLY PROHIBITED to use IDs in the queries and responses.
11. **Room or Object Specification:** Do not specify which room or object with a number (as with the ID or in full), for instance "bedroom 2" or "bed_3". Always use only the class or room name (e.g., "bedroom" or "bed").
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
    1. **Intent Fulfillment:** Does the obtained object/action satisfy the user's underlying need stated in the query?
    2. **Negative Constraint Matching (CRITICAL):** If the 'expected_answer' states the object is MISSING, and the 'obtained_response' ALSO states it is missing, this is a **True** match. Additional offers or suggestions (e.g., "Would you like something else?") do NOT degrade the score.
    3. **Action Equivalence:** Treat navigation verbs as synonymous (e.g., "Moving" == "Guiding").
    4. **Object Synonymy:** Treat standard English synonyms as the SAME Object Class (e.g., "Couch" == "Sofa").
    5. **Functional Equivalence:** If classes differ and are not synonyms, is it a valid proxy?
    6. **Ambiguity Check:** If the response is a question/list, does it correctly identify the target class present in the room?

    ### CASE 2: Multi-Turn (is_follow_up = True)
    Compare the conversational flow. Focus on whether the interaction is moving FORWARD:
    1. **Scope Narrowing:** Compare the robot's last question in 'obtained_messages' with its previous question. Did the robot incorporate the user's new constraint? If yes, this is **Partial** (Progressive Refinement).
    2. **Resolution Success:** Did the robot eventually identify the correct target?
    3. **Fallback Handling:** Did the robot offer a valid alternative if the specific target was missing?

</EVALUATION_LOGIC>

<SCORING_CRITERIA>
    Output one of the following verdicts:

    **True** (Strict Semantic Match / Action Equivalent / Correct Negative):
    - The target Object Class AND Room Type match the expectation exactly.
    - **Correct Negative:** The robot correctly identifies that the requested object is NOT in the area (matching the expected negative). Suggestions for other objects or general offers of help (e.g., "Would you like a coffee maker instead?") are acceptable and remain **True**.
    - The intent is captured and the action (or correct refusal) is executed.
    - If the target object is in a different room (e.g., Bathroom 1) and the robot correctly identifies the room name (e.g., Bathroom 2), the response is still considered True. For example, if the expected answer is "Taking you to the bed in Bedroom 1" and the obtained answer is "There is no bed in Bedroom 1, taking you to the bed in Bedroom 2", this should be considered True because the only difference is the room name, which can vary between the expected and obtained answers.
    - Hall, Hallway and Transitioning in our evaluation sceneraio must be considered the same, and by this reason, if the expected answer is "Taking you to the Hall" and the obtained answer is "Taking you to the Hallway" or "Transitioning", this should be considered True.

    **Partial** (Functional Success / Valid Alternative / Progressive Refinement):
    - **Functional Proxy:** The robot targets a **different** object class that logically supports the user's intent (when the target actually exists).
    - **Location Alternative / Fallback:** The robot offers a valid alternative of the same class in a different location (when the target exists elsewhere).
    - **Progressive Refinement:** The robot asks a follow-up question that is **more specific** than the previous one by applying the user's constraint (Multi-turn).
    - **Vagueness:** The robot identifies the correct object but omits the room name.
    - **Ambiguity Handling:** The robot correctly identifies the target Object Class and stops to ask for clarification.

    **False** (Failure):
    - **Functional Mismatch:** The robot targets an object that has NO utility for the user's query.
    - **False Positive:** The robot claims to find an object that the expected answer says is missing (Hallucination).
    - **Bad Negative:** The robot claims an object is missing when the expected answer says it exists.
    - **Stagnant Loop:** The robot asks the exact same question without incorporating new constraints.
</SCORING_CRITERIA>

<OUTPUT_FORMAT>
    Output ONLY one single word: True, Partial, or False.
</OUTPUT_FORMAT>
    """
)

ORIGINAL_JUDGE_PROMPT = dedent(
    """
<ROLE>
    You are an expert judge evaluating a semantic navigation system. Your goal is to compare the system's classification (`most_relevant_object` and `top_3_classes`) against the Ground Truth.
</ROLE>

<INPUT_DATA>
    Check the "is_follow_up" flag first.
    - If `false`: Analyze "query", "expected_answer", "most_relevant_object", "top_3_classes".
    - If `true`: Analyze "messages" (Ground Truth) and "obtained_messages" (System Output).
</INPUT_DATA>

<ID_HANDLING_RULE>
    **CRITICAL:** Strip all numerical suffixes/IDs (e.g., "bathroom_1" -> "bathroom"). Compare only base semantic Classes.
</ID_HANDLING_RULE>

<SYNONYM_RULES>
    Apply these equivalences STRICTLY before comparing targets. If A == B, treat them as the same class.
    - **General:** "None" == "null" == "Unknown" == "".
    - **Furniture:** "Couch" == "Sofa" == "Sofa Chair" == "Settee".
    - **Appliances:** "Tv" == "Television"; "Bin" == "Trash Can" == "Garbage".
    - **Rooms/Areas:** "Hall" == "Hallway" == "Transitioning".
</SYNONYM_RULES>

<LOGIC_ROUTER>
    IF "is_follow_up" is FALSE: GO TO [SECTION 1: SINGLE TURN EVALUATION].
    IF "is_follow_up" is TRUE:  GO TO [SECTION 2: MULTI-TURN EVALUATION].
</LOGIC_ROUTER>

<SECTION 1: SINGLE TURN EVALUATION>
    (Use this logic when is_follow_up = false)

    1. **Analyze Expectation:**
       - **Type A (Positive):** Expected answer implies object exists. Target = Object Name.
       - **Type B (Negative/Graceful Failure):** Expected answer implies object is MISSING. Target = NONE.

    2. **Verdict Logic:**
       - **CASE TYPE A (Positive):**
         - **True:** `most_relevant_object` matches Target (or synonym).
         - **Partial:** `most_relevant` misses, but Target (or synonym) is in `top_3_classes`.
         - **False:** Target not found.
       - **CASE TYPE B (Negative):**
         - **True:** `most_relevant_object` is NONE.
         - **Partial:** `most_relevant` is wrong, but NONE is in `top_3_classes`.
         - **False:** `most_relevant` is specific AND `top_3_classes` are all specific objects (Hallucination).
</SECTION 1: SINGLE TURN EVALUATION>

<SECTION 2: MULTI-TURN EVALUATION>
    (Use this logic when is_follow_up = true)

    **Context:** The conversation consists of 2 Robot Turns.
    - **Turn 1 (Ambiguity):** Robot should detect multiple instances of the object.
    - **Turn 2 (Resolution):** Robot should narrow down to the specific target.

    **Step 1: Extract Ground Truth Target**
    Identify the target object class from the `messages` (Ground Truth). Use <SYNONYM_RULES> to normalize (e.g., if User asks for "Sofa", Target is "Sofa" or "Couch").

    **Step 2: Evaluate Turn 1 (Ambiguity Phase)**
    Inspect the FIRST Robot response in `obtained_messages`.
    - **Condition T1-True:** `most_relevant_object` matches Target AND `top_3_classes` contains **MORE THAN 1** instance of Target/Synonyms (showing ambiguity detection).
    - **Condition T1-Partial:** `most_relevant_object` matches Target, BUT `top_3_classes` contains **ONLY 1** instance of Target (failed to see ambiguity, but saw the object).
    - **Condition T1-False:** Target is NOT in `most_relevant_object` AND NOT in `top_3_classes`.

    **Step 3: Evaluate Turn 2 (Resolution Phase)**
    Inspect the SECOND Robot response in `obtained_messages`.
    - **Condition T2-True:** `most_relevant_object` matches Target/Synonym.
    - **Condition T2-False:** Target/Synonym is NOT `most_relevant_object`.

    **Step 4: Final Aggregation**
    Combine verdicts from Turn 1 (T1) and Turn 2 (T2) using this PRIORITY table:

    1. **True:** IF (T1 is True) AND (T2 is True).
       *(Perfect execution: Saw duplicates, then picked the right one).*

    2. **Partial:** IF (T1 is True) AND (T2 is False).
       *(Valid Ambiguity: Correctly identified multiple candidates initially, but failed the final pick. Value provided).*

    3. **Partial:** IF (T1 is Partial) AND (T2 is True).
       *(Recovery: Missed the duplicates initially, but successfully navigated to the correct object in the end).*

    4. **False:** ALL OTHER COMBINATIONS.
       *(e.g., T1 is False - if it didn't see the object class initially, the rest is invalid).*
</SECTION 2: MULTI-TURN EVALUATION>

<OUTPUT_FORMAT>
    Output ONLY one single word: True, Partial, or False.
</OUTPUT_FORMAT>
    """
)
