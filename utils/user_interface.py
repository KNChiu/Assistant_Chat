class UserInterface:
    def __init__(self) -> None:
        pass

    def user_set(self, message: str, history: list) -> tuple:
        return "", history + [[message, ""]]

    def clear_chat(self) -> tuple:
        return "", "Send (0)"


if __name__ == "__main__":
    userinterface = UserInterface()

