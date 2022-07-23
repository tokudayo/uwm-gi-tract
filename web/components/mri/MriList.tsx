import { Col, Row } from "antd";
import MriItem from "./MriItem";

function MriList(props: any) {
  const users = {} as any;
  for (const mri of props.mris) {
    if (!users[mri.name]) users[mri.name] = [];
    users[mri.name].push(mri);
  }
  return (
    <>
      {Object.keys(users).map((name: any) => (
        <>
          <h1>{name}</h1>
            {users[name].map((mri: any) => (
              <>
                <MriItem
                  key={mri.id}
                  id={mri.id}
                  name={mri.name}
                  createdAt={mri.createdAt}
                  image0={mri.image0}
                  image1={mri.image1}
                  image2={mri.image2}
                />
                <br />
                <br />
                <br />
              </>
            ))}
        </>
      ))}
    </>
  );
}

export default MriList;
