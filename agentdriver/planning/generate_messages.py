def generate_messages(
    data_sample,
    use_peception=True,
    use_short_experience=True,
    verbose=True,
    use_gt_cot=False,
):
    token = data_sample["token"]
    ego = data_sample["ego"]
    perception = data_sample["perception"]
    commonsense = data_sample["commonsense"]
    experiences = data_sample["experiences"]
    reasoning = data_sample["reasoning"]
    long_experiences = (
        data_sample["long_experiences"] if "long_experiences" in data_sample else None
    )
    chain_of_thoughts = (
        data_sample["chain_of_thoughts"] if "chain_of_thoughts" in data_sample else ""
    )
    planning_target = (
        data_sample["planning_target"] if "planning_target" in data_sample else None
    )

    user_message = ego
    if use_peception:
        user_message += perception
    if use_short_experience:
        if experiences:
            user_message += experiences
    else:
        if long_experiences:
            user_message += long_experiences
    user_message += commonsense
    if use_gt_cot:
        user_message += chain_of_thoughts
    else:
        user_message += reasoning

    assistant_message = planning_target

    if verbose:
        print(user_message)
        print(assistant_message)

    return token, user_message, assistant_message
