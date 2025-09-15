def get_user_choice(prompt: str, options: list) -> str:
    """A helper function to get a valid choice from the user."""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")

    while True:
        try:
            choice_num = int(input(f"Enter your choice (1-{len(options)}): "))
            if 1 <= choice_num <= len(options):
                return options[choice_num - 1]

            else:
                print("Invalid choice. Please try again.")

        except ValueError:
            print("Invalid input. Please enter a number.")


def get_text_input(prompt: str, default: str = None) -> str:
    """A helper function to get a text input from the user."""
    prompt_text = f"{prompt} (default: {default}): " if default else f"{prompt}: "
    user_input = input(prompt_text).strip()
    return user_input or default