import { MongoClient } from "mongodb";
import { Kafka } from "kafkajs";
import { v4 as uuidv4 } from "uuid";

async function handler(req: any, res: any) {
  if (req.method === "POST") {
    const data = req.body;
    const id = uuidv4();
    console.log(data, 'Line #9 upload.ts');
    

    const client = await MongoClient.connect(process.env.MONGODB_URI as string);
    const db = client.db();

    const mrisCollection = db.collection("mris");

    const kafka = new Kafka({
      clientId: "web",
      brokers: [`${process.env.KAFKA_HOST}:${process.env.KAFKA_PORT}`],
    });
    const consumer = kafka.consumer({ groupId: "process.payload.reply" });
    const producer = kafka.producer();
    await producer.connect();
    await consumer.connect();
    await consumer.subscribe({
      topic: "process.payload.reply",
      fromBeginning: false,
    });
    await consumer.run({
      eachMessage: async ({ topic, message }) => {
        const resJSON = JSON.parse((message as any).value.toString());
        console.log(resJSON[3], 'Line #30 upload.ts');
        
        if (resJSON.length >= 4 && resJSON[3] == id) {
          await mrisCollection.insertOne({
            name: data.name,
            image0: resJSON[0],
            image1: resJSON[1],
            image2: resJSON[2],
            createdAt: Date.now(),
          });

          await client.close();
          await producer.disconnect();
          setImmediate(async () => {
            await consumer.disconnect();
          });

          res.status(201).json({ message: "MRI Uploaded!" });
        }
      },
    });

    await producer.send({
      topic: "process.payload",
      messages: [{ value: data.image, key: id }],
    });
  }
}

export default handler;
