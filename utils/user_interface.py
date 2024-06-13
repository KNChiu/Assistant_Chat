class UserInterface():
    def __init__(self) -> None:
        pass

    def change_prompt(self, choice: str) -> str:
        """
        Return different prompt messages based on the user's choice

        Parameters:
        choice (str): The user's choice

        Returns:
        str: The corresponding prompt message for the choice, respond in Traditional Chinese by default
        """
        if choice == "Code":
            return "你是一個生活在台灣的資深軟體工程師，使用 python 為主的程式語言，請根據提問生成合適的程式碼，並使用繁體中文條列說明功能"
        elif choice == "問答":
            return "你是一位人工智慧領域的專家，請專業並有邏輯的使用繁體中文回答問題"
        elif choice == "總結":
            return "你是一位重點統整的專家，請依據輸入的內容統整成簡短且有意義的文字，使用繁體中文回答"
        elif choice == "翻譯":
            return "依據輸入的文字，判斷是英文還是中文，如果輸入是英文翻譯成通順的中文，如果輸入是中文則翻譯成通順的英文"
        elif choice == "搜尋":
            return "依據 Google 搜尋結果統整資訊，並使用繁體中文簡短回答"
        else:
            return "使用繁體中文回應以下問題"

    def user_set(self, message, history):
        return "", history + [[message, ""]]

    def clear_chat(self):
        return "", "Send (0)"
