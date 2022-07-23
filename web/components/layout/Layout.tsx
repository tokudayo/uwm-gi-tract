import "antd/dist/antd.css";
import styles from "./Layout.module.css";
import { Layout as LayoutAnt, Menu } from "antd";
const { Header, Content, Footer } = LayoutAnt;
import { useRouter } from "next/router";
import { useEffect, useState } from "react";

function Layout(props: any) {
  const router = useRouter();

  return (
    <LayoutAnt className="layout">
      <Header>
        <div className={styles.logo} />
        <Menu
          theme="dark"
          mode="horizontal"
          items={[
            { key: 0, label: "Homepage" },
            { key: 1, label: "Upload Image" },
            { key: 2, label: "View Images" },
          ]}
          onClick={({ key }) => {
            const keyNum = Number(key);
            switch (keyNum) {
              case 0:
                router.push("/");
                break;
              case 1:
                router.push(`/upload`);
                break;
              case 2:
                router.push(`/images`);
                break;
            }
          }}
        />
      </Header>
      <Content
        style={{
          padding: "0 50px",
        }}
      >
        <div
          style={{
            margin: "16px 0",
          }}
        ></div>
        <div className={styles["site-layout-content"]}>{props.children}</div>
      </Content>
      <Footer
        style={{
          textAlign: "center",
        }}
      ></Footer>
    </LayoutAnt>
  );
}

export default Layout;
