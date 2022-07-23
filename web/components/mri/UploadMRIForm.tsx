import { Button, Form, Input, InputNumber, Select, Upload } from "antd";
import { InboxOutlined } from "@ant-design/icons";
import { useState } from "react";
const { Dragger } = Upload;
const { Option } = Select;
const layout = {
  labelCol: {
    span: 8,
  },
  wrapperCol: {
    span: 16,
  },
};
/* eslint-disable no-template-curly-in-string */

const validateMessages = {
  required: "${label} is required!",
  types: {
    email: "${label} is not a valid email!",
    number: "${label} is not a valid number!",
  },
  number: {
    range: "${label} must be between ${min} and ${max}",
  },
};

/* eslint-enable no-template-curly-in-string */

const UploadMRIForm = (props: any) => {
  const [fileLen, setFileLen] = useState(0);

  const onFinish = (values: any) => {
    props.onAddMRI(values);
  };

  const normFile = (e: any) => {
    console.log("Upload event:", e);

    if (Array.isArray(e)) {
      setFileLen(e.length);
      return e;
    }

    setFileLen(e?.fileList?.length);

    return e?.fileList;
  };

  return (
    <Form
      {...layout}
      name="nest-messages"
      onFinish={onFinish}
      validateMessages={validateMessages}
    >
      <Form.Item
        name="name"
        label="Full Name"
        rules={[
          {
            required: true,
          },
        ]}
      >
        <Input />
      </Form.Item>
      <Form.Item
        name="image"
        label="MRI Images"
        valuePropName="fileList"
        rules={[
          {
            required: true,
          },
          {
            validator: async () => {
              if (fileLen > 1) throw new Error("Single file only");
            },
            message: "Single file only",
          },
        ]}
        getValueFromEvent={normFile}
      >
        <Dragger name="image" multiple={false}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">
            Click or drag file to this area to upload
          </p>
        </Dragger>
      </Form.Item>
      <Form.Item wrapperCol={{ ...layout.wrapperCol, offset: 8 }}>
        <Button type="primary" htmlType="submit" loading={props.loading}>
          Submit
        </Button>
      </Form.Item>
    </Form>
  );
};

export default UploadMRIForm;