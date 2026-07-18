import sys
cars = [
    {
        "age": 2,
        "km": 30000,
        "engine": 9,
        "scratches": 1,
        "price": 190000
    },
    {
        "age": 4,
        "km": 70000,
        "engine": 8,
        "scratches": 3,
        "price": 155000
    },
    {
        "age": 6,
        "km": 120000,
        "engine": 7,
        "scratches": 5,
        "price": 120000
    },
    {
        "age": 8,
        "km": 180000,
        "engine": 6,
        "scratches": 8,
        "price": 90000
    },
    {
        "age": 10,
        "km": 250000,
        "engine": 5,
        "scratches": 10,
        "price": 65000
    }

]

most_expensive = cars[0]
most_less = cars[0]
total_km = 0
total_age = 0
total_price = 0
for car in cars:
    if car["price"] > most_expensive["price"]:
        most_expensive = car
    if car["price"] < most_less["price"]:
        most_less = car
    total_km += car["km"]
    total_age += car["age"]
    total_price += car["price"]


print("Most Expensive Car:")
print(most_expensive)
print("Most Affordable Car:")
print(most_less)
print(f"total Kilometers: {total_km}")
print(f"total Age: {total_age}")
print(f"total Price: {total_price}")
