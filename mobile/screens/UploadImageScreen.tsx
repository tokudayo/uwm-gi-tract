import { useEffect, useState } from "react";
import { StyleSheet, View, Text, Button } from "react-native";
import Input from "./Input";
// import { launchImageLibrary } from "react-native-image-picker";
import * as ImagePicker from "expo-image-picker";
import { useNavigation } from "@react-navigation/native";

/* eslint-enable no-template-curly-in-string */

async function onAddMri(navigation: any, inputs: any) {

  const response = await fetch("http://10.0.2.2:13008/api/upload", {
    method: "POST",
    body: JSON.stringify({
      name: inputs.name.value,
      image: `data:image/png;base64,${inputs.image.value}`,
    }),
    headers: {
      "Content-Type": "application/json",
    },
  });
  navigation.navigate("ViewImages"  as never);
}

const UploadImageScreen = (props: any) => {
  const navigation = useNavigation();

  const [inputs, setInputs] = useState({
    image: {
      value: null,
    },
    name: {
      value: "",
    },
  });

  function inputChangedHandler(inputIdentifier: string, enteredValue: any) {
    setInputs((curInputs) => {
      return {
        ...curInputs,
        [inputIdentifier]: { value: enteredValue },
      };
    });
  }

  function submitHandler() {
    onAddMri(navigation, inputs);
  }

  const pickImage = async () => {
    // No permissions request is necessary for launching the image library
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.All,
      quality: 1,
      base64: true,
    });

    if (!result.cancelled) {
      inputChangedHandler("image", result.base64);
    }
  };

  return (
    <View style={styles.container}>
      <Input
        label="Name"
        textInputConfig={{
          onChangeText: inputChangedHandler.bind(this, "name"),
          value: inputs.name.value,
        }}
      />
      <View style={styles.button}>
        <Button title="MRI Image" onPress={pickImage} />
      </View>
      <View style={styles.button}>
        <Button title="Submit" onPress={submitHandler} />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 15,
  },
  button: {
    marginVertical: 15,
  },
});

export default UploadImageScreen;
