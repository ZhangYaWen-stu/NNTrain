import json


def create_user():
    usernames = {}
    username = input("Please enter your username:")
    if username.lower() == "quit":
        print("Success quit.")
        return None
    info = {}
    password = input("Please enter your password:")
    if password.lower() == "quit":
        print("Success quit.")
        return None
    else:
        info["password"] = password
    name = input("Please enter your name:")
    if name.lower() == "quit":
        print("Success quit.")
        return None
    else:
        info["name"] = name
    phonenumber = input("Please enter your phone number:(Enter NO as not input)")
    if phonenumber.lower() == "quit":
        print("Success quit.")
        return None
    elif phonenumber.lower() == "no":
        pass
    else:
        info["phonenumber"] = phonenumber
    usernames[username] = info
    return usernames

def open_verify_info(filename):
    try:
        with open(filename) as file_ver:
            user_info = json.load(file_ver)
    except:
        print("You are using it for the first time, no user information file is found.(Enter quit to exit)")
        user_info = {}
    return user_info

def verify_user(filename):
    usernames = open_verify_info(filename)
    username = input("Please enter your username:")
    if username.lower() == "quit":
        print("Success quit.")
        return None
    if username in usernames.keys():
        password = input("Please enter your password:")
        while password != usernames[username]["password"]:
            print("Password error.(Enter quit to exit or continue)")
            password = input("Please enter your password again.")
            if password.lower() == "quit":
                print("Success quit.")
                return None
        return usernames[username]["name"]
    else:
        print("Not Found user, Creating new user......")
        username_new = create_user()
        if username_new:
            usernames[list(username_new.keys())[0]] = list(username_new.values())[0]
        else:
            print("Success quit.")
            return None
        with open(filename, 'w') as file_ver_new:
            json.dump(usernames, file_ver_new)
        return list(username_new.values())[0]["name"]

if __name__ == "__main__":
    filename = "file_\\verify_info.json"
    while True:
        name = verify_user(filename)
        if name:
            print("Welcome " + name + "!")
            break
        else:
            key = input("Do you want to continue?(yes/no)")
            if key.lower() == "yes":
                pass
            elif key.lower() == "no":
                print("Quit......")
                break
