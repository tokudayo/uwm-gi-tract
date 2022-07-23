import { Fragment, useEffect, useState } from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import UploadMRIForm from "../components/mri/UploadMRIForm";

function getBase64(file: any) {
  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = function () {
    console.log(reader.result);
  };
  reader.onerror = function (error) {
    console.log("Error: ", error);
  };
}

function UploadMRIPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  async function addMRIHandler(enteredMRIData: any) {
    setLoading(true);
    let reader = new FileReader();
    reader.onload = async(e: any) => {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: JSON.stringify({
          name: enteredMRIData.name,
          image: e.target.result,
        }),
        headers: {
          "Content-Type": "application/json",
        },
      });
      console.log(response);
      router.push(`/images`);
    };
    reader.readAsDataURL(enteredMRIData.image[0].originFileObj);
  }

  return (
    <Fragment>
      <UploadMRIForm onAddMRI={addMRIHandler} loading={loading} />
    </Fragment>
  );
}
export default UploadMRIPage;
