import { MongoClient } from "mongodb";
import { Kafka } from "kafkajs";
import { v4 as uuidv4 } from "uuid";

async function handler(req: any, res: any) {
  if (req.method === "GET") {
    // fetch data from an API
    const client = await MongoClient.connect(process.env.MONGODB_URI as string);

    const db = client.db();

    const mrisMri = db.collection("mris");

    const mris = await mrisMri.find().toArray();

    client.close();
    res.status(201).json({
      props: {
        mris: mris.map((mri) => ({
          ...mri,
          id: mri._id.toString(),
          _id: null,
        })),
      },
    });
  }
}

export default handler;
