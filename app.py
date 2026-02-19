import streamlit as st
import joblib
import pandas as pd

# --- PART 1: LOAD YOUR SAVED ML MODEL & SCALER ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('knn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please run your training script to create 'knn_model.pkl' and 'scaler.pkl'.")
        return None, None

knn_model, scaler = load_models()

# --- PART 2: YOUR INGREDIENT RECIPE LOGIC ---
def suggest_recipe(user_ingredients_str):
    available_ingredients = [ing.strip().lower() for ing in user_ingredients_str.split(',')]

    recipes = {
        "Pasta Primavera": {
            "ingredients": ["pasta", "assorted vegetables", "olive oil", "garlic", "parmesan cheese"],
            "instructions": "1. Cook pasta according to package instructions.\n2. In a pan, sauté assorted vegetables and minced garlic in olive oil until tender.\n3. Toss cooked pasta with the vegetable mixture.\n4. Serve hot, garnished with grated parmesan cheese."
        },
        "Vegetable Stir-fry": {
            "ingredients": ["assorted vegetables", "soy sauce", "sesame oil", "garlic", "ginger", "rice"],
            "instructions": "1. Cook rice according to package instructions.\n2. In a wok or skillet, stir-fry assorted vegetables, minced garlic, and ginger in sesame oil until tender.\n3. Add cooked rice and soy sauce, stir-fry until well combined.\n4. Serve hot."
        },
        "Caprese Salad": {
            "ingredients": ["tomatoes", "fresh mozzarella", "fresh basil leaves", "balsamic vinegar", "olive oil"],
            "instructions": "1. Slice tomatoes and fresh mozzarella cheese into rounds.\n2. Arrange alternating slices of tomatoes, mozzarella, and basil leaves on a plate.\n3. Drizzle with balsamic vinegar and olive oil.\n4. Serve chilled."
        },
        "Tomato Basil Pasta": {
            "ingredients": ["pasta", "tomatoes", "garlic", "fresh basil leaves", "olive oil"],
            "instructions": "1. Cook pasta according to package instructions.\n2. In a pan, sauté chopped tomatoes and minced garlic in olive oil until softened.\n3. Add fresh basil leaves and cook for another minute.\n4. Toss cooked pasta with the tomato-basil sauce.\n5. Serve hot."
        },
        "Egg Fried Rice": {
            "ingredients": ["rice", "eggs", "assorted vegetables", "soy sauce", "sesame oil", "garlic"],
            "instructions": "1. Cook rice according to package instructions and let it cool.\n2. In a wok or skillet, scramble eggs in sesame oil.\n3. Add minced garlic and assorted vegetables, stir-fry until tender.\n4. Add cooked rice and soy sauce, stir-fry until well combined.\n5. Serve hot."
        },
        "Chana Masala": {
            "ingredients": ["chickpeas", "onion", "tomatoes", "ginger", "garlic", "coriander", "cumin", "garam masala", "turmeric"],
            "instructions": "1. Soak chickpeas overnight and cook until soft.\n2. In a pan, sauté chopped onion, ginger, and garlic in oil until golden.\n3. Add chopped tomatoes and spices (coriander, cumin, garam masala, turmeric), cook until oil separates.\n4. Add cooked chickpeas and simmer for 10 minutes.\n5. Serve hot with rice or bread."
        },
        "Paneer Tikka Masala": {
            "ingredients": ["paneer", "onion", "bell peppers", "yogurt", "tomato puree", "ginger-garlic paste", "kasuri methi", "garam masala", "turmeric", "coriander powder", "cumin powder"],
            "instructions": "1. Marinate paneer cubes in yogurt, ginger-garlic paste, turmeric, and salt. Let it sit for 30 minutes.\n2. Grill or roast marinated paneer cubes until golden brown.\n3. In a pan, heat oil and add chopped onions. Cook until they turn translucent.\n4. Add ginger-garlic paste and sauté until the raw smell disappears.\n5. Add tomato puree, coriander powder, cumin powder, garam masala, and salt. Cook until the oil separates.\n6. Add grilled paneer cubes, bell peppers, and kasuri methi. Mix well.\n7. Cook for another 5-7 minutes.\n8. Garnish with cream and coriander leaves. Serve hot with naan or rice."
        },
        "Chicken Curry": {
            "ingredients": ["chicken", "onion", "tomatoes", "ginger", "garlic", "chilli powder", "turmeric", "coriander powder", "cumin powder", "garam masala", "coconut milk"],
            "instructions": "1. Heat oil in a pan and add chopped onions. Saute until golden brown.\n2. Add ginger and garlic paste, sauté until the raw smell disappears.\n3. Add chopped tomatoes and cook until they turn mushy.\n4. Add chili powder, turmeric, coriander powder, and cumin powder, garam masala, and salt. Cook until the oil separates.\n5. Add chicken pieces and coat well with the masala.\n6. Add water and cook until the chicken is tender.\n7. Add coconut milk and simmer for another 5-7 minutes.\n8. Garnish with coriander leaves. Serve hot with rice or roti."
        },
        "Palak Paneer": {
            "ingredients": ["paneer", "spinach", "onion", "tomato", "ginger", "garlic", "green chilies", "garam masala", "cumin", "coriander powder", "turmeric"],
            "instructions": "1. Blanch spinach leaves in boiling water for 2-3 minutes. Drain and blend into a smooth puree.\n2. Heat oil in a pan, add cumin seeds, chopped onions, ginger, garlic, and green chilies. Saute until onions turn golden brown.\n3. Add chopped tomatoes and cook until they soften.\n4. Add turmeric, coriander powder, and garam masala. Cook until the spices are fragrant.\n5. Pour in the spinach puree and mix well.\n6. Add cubed paneer and simmer for 5-7 minutes.\n7. Serve hot with roti or rice."
        },
        "Dal Tadka": {
            "ingredients": ["yellow lentils", "onion", "tomato", "ginger", "garlic", "green chilies", "turmeric", "cumin seeds", "mustard seeds", "asafoetida", "garam masala"],
            "instructions": "1. Rinse lentils and cook with water, turmeric, salt, and chopped tomatoes until soft. Mash slightly.\n2. Heat ghee in a pan, add cumin seeds, mustard seeds, asafoetida, chopped onions, ginger, garlic, and green chilies. Sauté until onions turn golden brown.\n3. Add cooked lentils and adjust consistency by adding water if needed.\n4. Add garam masala and simmer for a few minutes.\n5. Garnish with coriander leaves and serve hot with rice or roti."
        },
        "Aloo Gobi": {
            "ingredients": ["potatoes", "cauliflower", "onion", "tomato", "ginger", "garlic", "green chilies", "turmeric", "coriander powder", "cumin seeds", "garam masala"],
            "instructions": "1. Heat oil in a pan, add cumin seeds, chopped onions, ginger, garlic, and green chilies. Sauté until onions turn golden brown.\n2. Add chopped tomatoes and cook until they soften.\n3. Add turmeric, coriander powder, and garam masala. Cook until the spices are fragrant.\n4. Add diced potatoes and cauliflower florets. Mix well.\n5. Cover and cook until vegetables are tender.\n6. Garnish with coriander leaves and serve hot with roti or rice."
        },
        "Vegetable Biryani": {
            "ingredients": ["rice", "assorted vegetables", "onion", "tomato", "ginger", "garlic", "green chilies", "yogurt", "mint leaves", "coriander leaves", "garam masala", "turmeric", "saffron"],
            "instructions": "1. Soak rice in water for 30 minutes. Drain and set aside.\n2. Heat oil in a pan, add cumin seeds, chopped onions, ginger, garlic, and green chilies. Sauté until onions turn golden brown.\n3. Add assorted vegetables, yogurt, mint leaves, coriander leaves, garam masala, turmeric and saffron. Mix well.\n4. In a separate pot, layer soaked rice and vegetable mixture.\n5. Cover and cook on low heat until rice is cooked and aromatic.\n6. Serve hot with raita."
        },
        "Chicken Biryani": {
            "ingredients": ["rice", "chicken", "onion", "tomato", "ginger", "garlic", "green chilies", "yogurt", "mint leaves", "coriander leaves", "garam masala", "turmeric", "saffron"],
            "instructions": "1. Soak rice in water for 30 minutes. Drain and set aside.\n2. Marinate chicken pieces in yogurt, ginger-garlic paste, green chilies, turmeric, and garam masala. Let it sit for 30 minutes.\n3. Heat oil in a pan, add chopped onions and cook until golden brown.\n4. Add chopped tomatoes and cook until they soften.\n5. Add marinated chicken and cook until it is partially done.\n6. In a separate pot, layer soaked rice and chicken mixture. Add mint leaves, coriander leaves, and saffron.\n7. Cover and cook on low heat until rice is cooked and chicken is tender.\n8. Serve hot with raita."
        },
        "Pani Puri": {
            "ingredients": ["semolina flour", "all-purpose flour", "potatoes", "chickpeas", "tamarind pulp", "mint leaves", "green chilies", "chaat masala", "cumin powder", "coriander powder", "black salt"],
            "instructions": "1. Prepare a dough using semolina flour, all-purpose flour, and water. Let it rest for 30 minutes.\n2. Roll out small puris from the dough and deep fry until they puff up.\n3. Boil and mash potatoes, and prepare a spicy water by mixing tamarind pulp, mint leaves, green chilies, chaat masala, cumin powder, coriander powder, and black salt.\n4. Make a hole in each puri and stuff with mashed potatoes and chickpeas.\n5. Dip the stuffed puris in the spicy water and serve immediately."
        },
        "Masoor Dal": {
            "ingredients": ["red lentils", "onion", "tomato", "ginger", "garlic", "green chilies", "turmeric", "cumin seeds", "mustard seeds", "curry leaves", "asafoetida", "garam masala"],
            "instructions": "1. Rinse lentils and cook with water, turmeric, salt, and chopped tomatoes until soft. Mash slightly.\n2. Heat oil in a pan, add cumin seeds, mustard seeds, asafoetida, chopped onions, ginger, garlic, green chilies, and curry leaves. Saute until onions turn golden brown.\n3. Add cooked lentils and adjust consistency by adding water if needed.\n4. Add garam masala and simmer for a few minutes.\n5. Garnish with coriander leaves and serve hot with rice or roti."
        },
        "Vegetable Pakora": {
            "ingredients": ["gram flour", "assorted vegetables", "onion", "ginger", "garlic", "green chilies", "coriander leaves", "cumin seeds", "carom seeds", "turmeric", "chili powder", "garam masala"],
            "instructions": "1. Mix gram flour, chopped onions, ginger, garlic, green chilies, coriander leaves, cumin seeds, carom seeds, turmeric, chili powder, garam masala, and salt in a bowl. Add water to make a thick batter.\n2. Heat oil in a deep frying pan. Drop spoonfuls of the batter into the hot oil and fry until golden brown and crispy.\n3. Serve hot with mint chutney or tamarind chutney."
        }
    }

    threshold = 0.6
    suggested_recipes_list = []

    for recipe, details in recipes.items():
        required_ingredients = details["ingredients"]
        matching_ingredients = sum(ingredient in available_ingredients for ingredient in required_ingredients)
        if len(required_ingredients) > 0 and (matching_ingredients / len(required_ingredients)) >= threshold:
            suggested_recipes_list.append((recipe, details["instructions"]))
    
    return suggested_recipes_list

# --- PART 3: SIMULATED PRICE COMPARISON LOGIC ---
def get_simulated_prices(craving):
    price_db = {
        "Biryani": ("₹250.00", "₹240.00"),
        "Pizza": ("₹300.00", "₹310.00"),
        "Dosa": ("₹80.00", "₹85.00"),
        "Paneer Butter Masala": ("₹220.00", "₹230.00"),
        "Ice cream": ("₹100.00", "₹95.00"),
        "Pasta Primavera": ("₹180.00", "₹170.00"),
        "Vegetable Stir-fry": ("₹150.00", "₹145.00"),
        "Caprese Salad": ("₹200.00", "₹190.00"),
        "Tomato Basil Pasta": ("₹160.00", "₹155.00"),
        "Egg Fried Rice": ("₹130.00", "₹125.00"),
        "Chana Masala": ("₹140.00", "₹135.00"),
        "Paneer Tikka Masala": ("₹230.00", "₹225.00"),
        "Chicken Curry": ("₹260.00", "₹255.00"),
        "Palak Paneer": ("₹210.00", "₹205.00"),
        "Dal Tadka": ("₹120.00", "₹115.00"),
        "Aloo Gobi": ("₹130.00", "₹125.00"),
        "Vegetable Biryani": ("₹200.00", "₹195.00"),
        "Pani Puri": ("₹60.00", "₹55.00"),
        "Masoor Dal": ("₹110.00", "₹105.00"),
        "Vegetable Pakora": ("₹90.00", "₹85.00")
    }
    return price_db.get(craving, ("N/A", "N/A"))

# --- PART 4: THE STREAMLIT USER INTERFACE (UI) ---
st.title("🍽️ Foodie Saver")
st.caption("Finding the Best Bite for your Buck")

menu = st.sidebar.selectbox("Choose Your Feature", ["Mood-Based Meals", "Ingredient Recipes", "Price Comparison"])

if menu == "Mood-Based Meals":
    st.header("😊 Mood-Based Meal Suggestions")
    col1, col2 = st.columns(2)
    with col1:
        user_age = st.number_input("Enter your age:", min_value=10, max_value=80, value=25)
        user_gender = st.selectbox("Your gender:", ("MALE", "FEMALE", "OTHERS"))
    with col2:
        user_time_of_day = st.selectbox("Time of the day:", ("MORNING", "AFTERNOON", "EVENING", "NIGHT"))
        user_mood = st.selectbox("Your mood:", ("POSITIVE", "NEGATIVE", "NEUTRAL"))

    if st.button("Suggest My Craving"):
        if knn_model and scaler:
            if user_age < 10:
                age_range_str = '<10'
            elif user_age <= 20:
                age_range_str = '10-20'
            elif user_age <= 30:
                age_range_str = '20-30'
            elif user_age <= 40:
                age_range_str = '30-40'
            elif user_age <= 50:
                age_range_str = '40-50'
            elif user_age <= 60:
                age_range_str = '50-60'
            else:
                age_range_str = '>60'

            age_mapping = {'<10': 0, '10-20': 1, '20-30': 2, '30-40': 3, '40-50': 4, '50-60': 5, '>60': 6}
            user_age_range = age_mapping[age_range_str]

            gender_mapping = {'MALE': 0, 'FEMALE': 1, 'OTHERS': 2}
            time_of_day_mapping = {'MORNING': 0, 'AFTERNOON': 1, 'EVENING': 2, 'NIGHT': 3}
            mood_mapping = {'POSITIVE': 0, 'NEGATIVE': 1, 'NEUTRAL': 2}

            user_input = [[
                user_age_range,
                gender_mapping[user_gender],
                time_of_day_mapping[user_time_of_day],
                mood_mapping[user_mood]
            ]]

            user_input_scaled = scaler.transform(user_input)
            predicted_craving = knn_model.predict(user_input_scaled)[0]

            st.success(f"### Predicted craving: {predicted_craving}!")
            swiggy_price, zomato_price = get_simulated_prices(predicted_craving)
            st.subheader("Price Comparison:")
            st.markdown(f"**Swiggy:** `{swiggy_price}`")
            st.markdown(f"**Zomato:** `{zomato_price}`")
        else:
            st.error("Model is not loaded. Check Step 1.")

elif menu == "Ingredient Recipes":
    st.header("🍳 Ingredient-Based Recipe Creation")
    ingredients_input = st.text_input("Enter ingredients (comma-separated):", placeholder="rice, chicken, onion")
    if st.button("Generate Recipe"):
        if ingredients_input:
            recipes_found = suggest_recipe(ingredients_input)
            if recipes_found:
                st.subheader(f"Found {len(recipes_found)} matching recipe(s):")
                for recipe, instructions in recipes_found:
                    with st.expander(f"**{recipe}**"):
                        st.text(instructions)
            else:
                st.warning("No recipe found. Try adding more ingredients!")
        else:
            st.error("Please enter at least one ingredient.")

elif menu == "Price Comparison":
    st.header("💸 Price Comparison (Simulation)")
    food_to_compare = st.selectbox("Select a food to compare:", 
        ("Biryani", "Pizza", "Dosa", "Paneer Butter Masala", "Ice cream", "Pasta Primavera", "Vegetable Stir-fry", "Caprese Salad", "Tomato Basil Pasta", "Egg Fried Rice", "Chana Masala", "Paneer Tikka Masala", "Chicken Curry", "Palak Paneer", "Dal Tadka", "Aloo Gobi", "Vegetable Biryani", "Pani Puri", "Masoor Dal", "Vegetable Pakora"))
    swiggy_price, zomato_price = get_simulated_prices(food_to_compare)
    df = pd.DataFrame({
        "Platform": ["Swiggy", "Zomato"],
        "Simulated Price (₹)": [swiggy_price, zomato_price],
        "Delivery Time": ["25-30 mins", "30-35 mins"]
    })
    st.table(df)