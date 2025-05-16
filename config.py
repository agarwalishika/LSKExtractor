translated_generation_paths = "mlk_gens/"
LANGUAGE_SET = ["Chinese", "English", "French", "Spanish", "Portuguese", "German", "Italian", "Russian", "Japanese", "Korean", "Vietnamese", "Thai", "Arabic"]

CultureAtlas_LANGUAGE_SET_REASONING = {
    "English": lambda input: f"""Is the following correct? {input['input']}
Country: {input['country']}, Topic: {input['topic']}, Subtopic: {input['subtopic']}
Think about and answer in English, and then output our final answer as True or False. Fill out the following JSON:
{{
    "reasoning_in_English": "<your reasoning steps in English>",
    "final_answer": "<output answer here>"
}}""",

    "Chinese": lambda input: f"""以下内容正确吗？ {input['input']}
国家：{input['country']}，主题：{input['topic']}，子主题：{input['subtopic']}
请用中文思考并回答，然后将我们的最终答案输出为 True 或 False。请按以下 JSON 格式填写：
{{
    "reasoning_in_Chinese": "<你的中文推理步骤>",
    "final_answer": "<在此处填写你的最终答案>"
}}""",

    "French": lambda input: f"""Ce qui suit est-il correct ? {input['input']}
Pays : {input['country']}, Sujet : {input['topic']}, Sous-sujet : {input['subtopic']}
Réfléchissez et répondez en français, puis renvoyez votre réponse finale sous forme de True ou False. Veuillez remplir le JSON suivant :
{{
    "reasoning_in_French": "<vos étapes de raisonnement en français>",
    "final_answer": "<votre réponse finale ici>"
}}""",

    "Spanish": lambda input: f"""¿Es lo siguiente correcto? {input['input']}
País: {input['country']}, Tema: {input['topic']}, Subtema: {input['subtopic']}
Piensa y responde en español, luego proporciona tu respuesta final como True o False. Por favor, completa el siguiente JSON:
{{
    "reasoning_in_Spanish": "<tus pasos de razonamiento en español>",
    "final_answer": "<tu respuesta final aquí>"
}}""",

    "Portuguese": lambda input: f"""O seguinte está correto? {input['input']}
País: {input['country']}, Tópico: {input['topic']}, Subtópico: {input['subtopic']}
Pense e responda em português, e depois forneça sua resposta final como True ou False. Por favor, preencha o JSON a seguir:
{{
    "reasoning_in_Portuguese": "<seus passos de raciocínio em português>",
    "final_answer": "<sua resposta final aqui>"
}}""",

    "German": lambda input: f"""Ist Folgendes korrekt? {input['input']}
Land: {input['country']}, Thema: {input['topic']}, Unterthema: {input['subtopic']}
Denken Sie auf Deutsch nach und antworten Sie auf Deutsch, und geben Sie Ihre endgültige Antwort als True oder False aus. Bitte füllen Sie das folgende JSON aus:
{{
    "reasoning_in_German": "<Ihre Begründungsschritte auf Deutsch>",
    "final_answer": "<Ihre endgültige Antwort hier>"
}}""",

    "Italian": lambda input: f"""La seguente affermazione è corretta? {input['input']}
Paese: {input['country']}, Argomento: {input['topic']}, Sottotema: {input['subtopic']}
Rifletti e rispondi in italiano, quindi fornisci la tua risposta finale come True o False. Per favore, compila il seguente JSON:
{{
    "reasoning_in_Italian": "<i tuoi passaggi di ragionamento in italiano>",
    "final_answer": "<la tua risposta finale qui>"
}}""",

    "Russian": lambda input: f"""Следующее утверждение верно? {input['input']}
Страна: {input['country']}, Тема: {input['topic']}, Подтема: {input['subtopic']}
Обдумайте и ответьте на русском, затем выведите окончательный ответ как True или False. Пожалуйста, заполните следующий JSON:
{{
    "reasoning_in_Russian": "<ваши шаги рассуждения на русском>",
    "final_answer": "<ваш окончательный ответ здесь>"
}}""",

    "Japanese": lambda input: f"""次の内容は正しいですか？ {input['input']}
国：{input['country']}、トピック：{input['topic']}、サブトピック：{input['subtopic']}
日本語で考えて回答し、その後最終回答を True または False で出力してください。次の JSON を埋めてください：
{{
    "reasoning_in_Japanese": "<あなたの日本語での思考ステップ>",
    "final_answer": "<最終回答をここに入力>"
}}""",

    "Korean": lambda input: f"""다음 내용이 올바른가요? {input['input']}
국가: {input['country']}, 주제: {input['topic']}, 하위 주제: {input['subtopic']}
한국어로 생각하고 답변한 다음 최종 답변을 True 또는 False로 출력하세요. 다음 JSON을 채워 주세요:
{{
    "reasoning_in_Korean": "<한국어로 된 추론 단계>",
    "final_answer": "<최종 답변을 여기에 입력>"
}}""",

    "Vietnamese": lambda input: f"""Đoạn sau có đúng không? {input['input']}
Quốc gia: {input['country']}, Chủ đề: {input['topic']}, Chủ đề phụ: {input['subtopic']}
Hãy suy nghĩ và trả lời bằng tiếng Việt, sau đó xuất kết quả cuối cùng là True hoặc False. Vui lòng điền vào JSON sau:
{{
    "reasoning_in_Vietnamese": "<các bước lập luận bằng tiếng Việt của bạn>",
    "final_answer": "<câu trả lời cuối cùng của bạn tại đây>"
}}""",

    "Thai": lambda input: f"""เนื้อหาต่อไปนี้ถูกต้องหรือไม่? {input['input']}
ประเทศ: {input['country']}, หัวข้อ: {input['topic']}, หัวข้อย่อย: {input['subtopic']}
โปรดคิดและตอบเป็นภาษาไทย จากนั้นให้ส่งคำตอบสุดท้ายเป็น True หรือ False กรุณากรอก JSON ดังนี้:
{{
    "reasoning_in_Thai": "<ขั้นตอนการวิเคราะห์ของคุณเป็นภาษาไทย>",
    "final_answer": "<คำตอบสุดท้ายของคุณที่นี่>"
}}""",

    "Arabic": lambda input: f"""هل ما يلي صحيح؟ {input['input']}
البلد: {input['country']}, الموضوع: {input['topic']}, الموضوع الفرعي: {input['subtopic']}
فكر وأجب باللغة العربية، ثم أخرج إجابتك النهائية كـ True أو False. يرجى ملء JSON التالي:
{{
    "reasoning_in_Arabic": "<خطواتك في التفكير باللغة العربية>",
    "final_answer": "<إجابتك النهائية هنا>"
}}""",
}



MCQ_LANGUAGE_SET_REASONING = {
    "English": lambda input: f"""Question: {input['input']}
Answer choices: {input['choices']}.
Think about it in English, and then select one of the answer choices. Fill out the following JSON:
{{
    "reasoning_in_English": "<your reasoning steps in English>",
    "final_answer": "<output answer here>"
}}
""",

    "Chinese": lambda input: f"""问题：{input['input']}
答案选项：{input['choices']}。
请用中文思考，然后从答案选项中选择一个。请按以下 JSON 格式填写：
{{
    "reasoning_in_Chinese": "<你的中文推理步骤>",
    "final_answer": "<在此处填写你的答案>"
}}
""",

    "French": lambda input: f"""Question : {input['input']}
Options de réponse : {input['choices']}.
Réfléchissez en français, puis sélectionnez l'une des options de réponse. Veuillez remplir le JSON suivant :
{{
    "reasoning_in_French": "<vos étapes de raisonnement en français>",
    "final_answer": "<votre réponse finale ici>"
}}
""",

    "Spanish": lambda input: f"""Pregunta: {input['input']}
Opciones de respuesta: {input['choices']}.
Piensa en español y luego selecciona una de las opciones de respuesta. Completa el siguiente JSON:
{{
    "reasoning_in_Spanish": "<tus pasos de razonamiento en español>",
    "final_answer": "<tu respuesta final aquí>"
}}
""",

    "Portuguese": lambda input: f"""Pergunta: {input['input']}
Opções de resposta: {input['choices']}.
Pense em português e, em seguida, selecione uma das opções de resposta. Preencha o seguinte JSON:
{{
    "reasoning_in_Portuguese": "<seus passos de raciocínio em português>",
    "final_answer": "<sua resposta final aqui>"
}}
""",

    "German": lambda input: f"""Frage: {input['input']}
Antwortmöglichkeiten: {input['choices']}.
Denken Sie auf Deutsch nach und wählen Sie dann eine der Antwortmöglichkeiten aus. Füllen Sie das folgende JSON aus:
{{
    "reasoning_in_German": "<Ihre Begründungsschritte auf Deutsch>",
    "final_answer": "<Ihre endgültige Antwort hier>"
}}
""",

    "Italian": lambda input: f"""Domanda: {input['input']}
Scelte di risposta: {input['choices']}.
Rifletti in italiano, quindi seleziona una delle opzioni di risposta. Compila il seguente JSON:
{{
    "reasoning_in_Italian": "<i tuoi passaggi di ragionamento in italiano>",
    "final_answer": "<la tua risposta finale qui>"
}}
""",

    "Russian": lambda input: f"""Вопрос: {input['input']}
Варианты ответа: {input['choices']}.
Обдумайте на русском, а затем выберите один из вариантов ответа. Заполните следующий JSON:
{{
    "reasoning_in_Russian": "<ваши шаги рассуждения на русском>",
    "final_answer": "<ваш окончательный ответ здесь>"
}}
""",

    "Japanese": lambda input: f"""質問: {input['input']}
回答選択肢: {input['choices']}。
日本語で考え、その後回答の選択肢の中からひとつを選んでください。次のJSONを埋めてください：
{{
    "reasoning_in_Japanese": "<あなたの日本語での思考ステップ>",
    "final_answer": "<最終的な回答をここに記入>"
}}
""",

    "Korean": lambda input: f"""질문: {input['input']}
답변 선택지: {input['choices']}.
한국어로 생각한 후 답변 선택지 중 하나를 선택하세요. 다음 JSON을 작성하세요:
{{
    "reasoning_in_Korean": "<한국어로 된 추론 단계>",
    "final_answer": "<최종 답변을 여기에 입력>"
}}
""",

    "Vietnamese": lambda input: f"""Câu hỏi: {input['input']}
Các lựa chọn trả lời: {input['choices']}.
Hãy suy nghĩ bằng tiếng Việt, sau đó chọn một trong các lựa chọn trả lời. Điền vào JSON sau:
{{
    "reasoning_in_Vietnamese": "<các bước lập luận bằng tiếng Việt của bạn>",
    "final_answer": "<câu trả lời cuối cùng của bạn tại đây>"
}}
""",

    "Thai": lambda input: f"""คำถาม: {input['input']}
ตัวเลือกคำตอบ: {input['choices']}.
โปรดคิดเป็นภาษาไทย แล้วเลือกคำตอบหนึ่งในตัวเลือก จากนั้นกรอก JSON ดังนี้:
{{
    "reasoning_in_Thai": "<ขั้นตอนการวิเคราะห์ของคุณเป็นภาษาไทย>",
    "final_answer": "<คำตอบสุดท้ายของคุณที่นี่>"
}}
""",

    "Arabic": lambda input: f"""سؤال: {input['input']}
خيارات الإجابة: {input['choices']}.
فكر باللغة العربية ثم اختر أحد خيارات الإجابة. املأ JSON التالي:
{{
    "reasoning_in_Arabic": "<خطواتك في التفكير باللغة العربية>",
    "final_answer": "<إجابتك النهائية هنا>"
}}
""",
}



datasets = [
    "CultureAtlas",
    "BLEnD",
    "SocialIQA"
]

model_names = [
    "google/gemma-3-1b-it",                     # 1b
    "google/gemma-3-9b-it",                    # 12b
    "meta-llama/Llama-3.2-1B-Instruct",         # 1b
    "meta-llama/Llama-3.2-3B-Instruct",         # 3b
    "meta-llama/Llama-3.1-8B-Instruct",         # 8b
    "ibm-granite/granite-3.1-8b-instruct",      # 8b
    "microsoft/Phi-3-small-8k-instruct",        # 7b
    "microsoft/Phi-3-medium-128k-instruct",     # 14b
    "CohereLabs/aya-23-8B",                     # 8b
    "CohereLabs/aya-23-35B",                    # 35b
]