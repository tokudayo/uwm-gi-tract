import { Card, Col, Image, Row } from "antd";
import Link from "next/link";
const { Meta } = Card;

function MriItemItem(props: any) {
  return (
    <>
      <p>{new Date(props.createdAt).toLocaleString()}</p>
      <Row>
        <Image.PreviewGroup>
          <Col className="gutter-row" span={8}>
            <Image
              alt="example"
              width={400}
              src={`data:image/png;base64,${props.image0}`}
            />
          </Col>
          <Col className="gutter-row" span={8}>
            <Image
              alt="example"
              width={400}
              src={`data:image/png;base64,${props.image1}`}
            />
          </Col>
          <Col className="gutter-row" span={8}>
            <Image
              alt="example"
              width={400}
              src={`data:image/png;base64,${props.image2}`}
            />
          </Col>
        </Image.PreviewGroup>
      </Row>
    </>
  );
}

export default MriItemItem;
