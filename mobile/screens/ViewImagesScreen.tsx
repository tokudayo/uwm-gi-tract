import { Card } from "@rneui/themed";
import { useEffect, useState } from "react";
import { FlatList, RefreshControl, StyleSheet, View } from "react-native";

function Image(props: any) {
  return (
    <>
      <Card>
        <Card.Image
          // style={{ width: 300, height: 300 }}
          source={{
            uri: `data:image/png;base64,${props.image0}`,
          }}
        />
        <Card.Title>{`${props.name} - ${new Date(props.createdAt).toLocaleString()} - 0`}</Card.Title>
      </Card>
      <Card>
        <Card.Image
          // style={{ width: 300, height: 300 }}
          source={{
            uri: `data:image/png;base64,${props.image1}`,
          }}
        />
        <Card.Title>{`${props.name} - ${new Date(props.createdAt).toLocaleString()} - 1`}</Card.Title>
      </Card>
      <Card>
        <Card.Image
          // style={{ width: 300, height: 300 }}
          source={{
            uri: `data:image/png;base64,${props.image2}`,
          }}
        />
        <Card.Title>{`${props.name} - ${new Date(props.createdAt).toLocaleString()} - 2`}</Card.Title>
      </Card>
    </>
  );
}

function ViewImages() {
  const [res, setRes] = useState(undefined as any);
  useEffect(() => {
    const callAPI = async() => {
      const res = await fetch("http://10.0.2.2:13008/api/images");
      const json = await res.json();
      setRes(json);
    }
    callAPI();
  }, []);

  const renderItem = ({ item }: any) => (
    <Image
      key={item.id}
      id={item.id}
      name={item.name}
      createdAt={item.createdAt}
      image0={item.image0}
      image1={item.image1}
      image2={item.image2}
    />
  );

  return (
    <View style={[styles.container]}>
      {res && <FlatList
        data={res.props.mris}
        renderItem={renderItem}
        keyExtractor={(item) => item.id}
      />}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 15,
    justifyContent: "center",
  },
  button: {
    marginVertical: 15,
  },
});

export default ViewImages;
