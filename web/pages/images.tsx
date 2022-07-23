import type { NextPage } from "next";
import { MongoClient } from "mongodb";
import MriList from "../components/mri/MriList";

const Mris: NextPage = (props: any) => {
  return <MriList mris={props.mris} />;
};

export async function getServerSideProps() {
  // fetch data from an API
  const client = await MongoClient.connect(process.env.MONGODB_URI as string);

  const db = client.db();

  const mrisMri = db.collection("mris");

  const mris = await mrisMri.find().toArray();

  client.close();

  return {
    props: {
      mris: mris.map((mri) => ({
        ...mri,
        id: mri._id.toString(),
        _id: null,
      })),
    },
  };
}

export default Mris;
