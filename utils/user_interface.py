class UserInterface:
    def __init__(self, prompt_setting) -> None:
        self.prompt_setting = prompt_setting

    def change_prompt(self, choice: str) -> str:
        return self.prompt_setting.get(choice, "使用繁體中文回應以下問題")

    def user_set(self, message: str, history: list) -> tuple:
        return "", history + [[message, ""]]

    def clear_chat(self) -> tuple:
        return "", "Send (0)"


if __name__ == "__main__":
    prompt_setting = {
        "Code": "你是一個生活在台灣的資深軟體工程師，使用 python 為主的程式語言，請根據提問生成合適的程式碼，並使用繁體中文條列說明功能",
        "問答": "你是一位人工智慧領域的專家，請專業並有邏輯的使用繁體中文回答問題",
        "總結": "你是一位重點統整的專家，請依據輸入的內容統整成簡短且有意義的文字，使用繁體中文回答",
        "翻譯": "依據輸入的文字，判斷是英文還是繁體中文，如果輸入是英文翻譯成通順的中文，如果輸入是中文則翻譯成通順的英文",
        "搜尋": "依據 Google 搜尋結果統整資訊，並使用繁體中文簡短回答",
    }

    userinterface = UserInterface(prompt_setting)
    print(userinterface.change_prompt(choice="Code"))
