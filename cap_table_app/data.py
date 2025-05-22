import dspy

# Example training data for optimizing the cap table formatter.
# Each Example records an input text, the founders list, and the expected formatted output.

trainset = [
    dspy.Example(
        cap_table_answer="Jane owns 60% of the company while John has 30%. The remaining 10% is reserved for the employee option pool.",
        founders_list="Jane Smith, John Doe",
        cap_table={
            "founders": [
                {"full_name": "Jane Smith", "ownership_percentage": 60},
                {"full_name": "John Doe", "ownership_percentage": 30},
            ],
            "ownership_groups": [
                {"group_name": "ESOP", "ownership_percentage": 10},
            ],
        },
    ).with_inputs("cap_table_answer", "founders_list"),
    dspy.Example(
        cap_table_answer="Michael and Sarah each own 45% of the company. Robert left last year and doesn't own any equity. There's a 10% option pool for employees.",
        founders_list="Michael Johnson, Sarah Williams, Robert Chen",
        cap_table={
            "founders": [
                {"full_name": "Michael Johnson", "ownership_percentage": 45},
                {"full_name": "Sarah Williams", "ownership_percentage": 45},
                {"full_name": "Robert Chen", "ownership_percentage": 0},
            ],
            "ownership_groups": [
                {"group_name": "ESOP", "ownership_percentage": 10},
            ],
        },
    ).with_inputs("cap_table_answer", "founders_list"),
    dspy.Example(
        cap_table_answer="Based on the transcript, this information was not discussed or disclosed.",
        founders_list="Alex Brown, Lisa Garcia",
        cap_table={
            "founders": [
                {"full_name": "Alex Brown", "ownership_percentage": None},
                {"full_name": "Lisa Garcia", "ownership_percentage": None},
            ],
            "ownership_groups": [],
        },
    ).with_inputs("cap_table_answer", "founders_list"),
    dspy.Example(
        cap_table_answer="According to the call the two founders currently split the common equity equally—50 percent each—because they believe in equality of outcomes. No other equity holders or advisors have been added so far, but a 10 percent option pool has been set aside and not yet granted. Outside capital consists solely of a16z Speedrun Programme SAFEs: US $500 000 on a US $5 million valuation cap and US $250 000 on an uncapped SAFE that will convert in the next priced round. Those SAFEs are still unconverted, so the precise post-money percentages were not discussed.",
        founders_list="Ashish Ranjan Jha, Ahmed Ahres",
        cap_table={
            "founders": [
                {"full_name": "Ashish Ranjan Jha", "ownership_percentage": 45},
                {"full_name": "Ahmed Ahres", "ownership_percentage": 45},
            ],
            "ownership_groups": [
                {"group_name": "ESOP", "ownership_percentage": 10},
            ],
        },
    ).with_inputs("cap_table_answer", "founders_list"),
    dspy.Example(
        cap_table_answer="Ownership is split almost evenly between the two founders, with one holding 51 % and the other 49 %. No outside investors, option pool, advisor shares, or other equity holders were mentioned, so the cap table currently consists solely of the two founders in roughly equal proportions.",
        founders_list="Arda Altug, Ali Kababiyik",
        cap_table={
            "founders": [
                {"full_name": "Arda Altug", "ownership_percentage": 51},
                {"full_name": "Ali Kababiyik", "ownership_percentage": 49},
            ],
            "ownership_groups": [],
        },
    ).with_inputs("cap_table_answer", "founders_list"),
    dspy.Example(
        cap_table_answer="Only the founders' equity and an unallocated pool were disclosed. Jawwad said he holds approximately 51 % but later noted the exact figure may have settled closer to about 47 %. Anoop initially was cited at 39 % and then described as a little bit less after the final adjustments. Roughly 10 % of the cap is being kept open for future purposes (option pool, additional early team, or similar). All outside money to date—US $250 k raised on SAFEs—has not yet converted, so those instruments currently represent no cap-table percentage. No other shareholders, advisors, or pools were mentioned.",
        founders_list="Jawwad Rafique, Anoop Dixith",
        cap_table={
            "founders": [
                {"full_name": "Jawwad Rafique", "ownership_percentage": 51},
                {"full_name": "Anoop Dixith", "ownership_percentage": 39},
            ],
            "ownership_groups": [
                {"group_name": "ESOP", "ownership_percentage": 10},
            ],
        },
    ).with_inputs("cap_table_answer", "founders_list"),
    dspy.Example(
        cap_table_answer="Among the founders Ahmed owns approximately 85 percent of the company and Brandon owns roughly 15 percent. In addition to the founders' stakes, a pool of equity amounting to less than 14–15 percent in total has been granted to early employees and named advisors, including the CTO Andrew Clark, lead engineer Austin Birch, and several industry advisors (Andrew Casey, Kevin O'Neil, Bo Davis, and Brad Garfield). No exact option-pool size, investor ownership, or other specific percentages were disclosed beyond these figures.",
        founders_list="Ahmed Nawash, Brandon O'Connor",
        cap_table={
            "founders": [
                {"full_name": "Ahmed Nawash", "ownership_percentage": 73.1},
                {"full_name": "Brandon O'Connor", "ownership_percentage": 12.9},
            ],
            "ownership_groups": [
                {"group_name": "ESOP & Advisors", "ownership_percentage": 14},
            ],
        },
    ).with_inputs("cap_table_answer", "founders_list"),
]

devset = trainset  # For demonstration purposes
