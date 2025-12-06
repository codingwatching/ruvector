import { RuVectorHome } from 'components/interfaces/RuVector/RuVectorHome'
import { ProjectLayoutWithAuth } from 'components/layouts/ProjectLayout/ProjectLayout'
import type { NextPageWithLayout } from 'types'

const HomePage: NextPageWithLayout = () => {
  return <RuVectorHome />
}

HomePage.getLayout = (page) => <ProjectLayoutWithAuth>{page}</ProjectLayoutWithAuth>

export default HomePage
