"""

Config file for Streamlit App

"""

from member import Member


TITLE = "My Awesome App"

TEAM_MEMBERS = [
    Member(
        name="Damien BRUZZONE",
        linkedin_url="https://www.linkedin.com/in/damien-bruzzone-7b2836148/"
    ),
    Member(name="Nathan CEBRON",
        linkedin_url="https://www.linkedin.com/in/nathan-cebron-9992a1a4/"),
    Member(name="Matthieu PELINGRE",
        linkedin_url="https://www.linkedin.com/in/matthieu-pelingre-3667b0210/",
)]

PROMOTION = "Promotion Bootcamp Data Scientist - Mai 2022"

MENTOR = "Gaspard GRIMM"
